// SPDX-License-Identifier: Apache-2.0
// Copyright (c) NVIDIA CORPORATION

use ::NVRC::daemon::{FABRIC_MODE_FULL, FABRIC_MODE_SHARED};
use ::NVRC::kata_agent::{self, SYSLOG_POLL_FOREVER as POLL_FOREVER};
use ::NVRC::nvrc::NVRC;
use ::NVRC::toolkit::nvidia_ctk_cdi;
use ::NVRC::{
    gpu_extension, guest_extension_image, hash, infiniband, init, kmsg, lockdown, mode, modprobe,
    mount, net, syslog, time,
};
use log::info;

/// Every signal that tells whether nvidia.ko initialized at least one GPU,
/// logged after modprobe so a failing run carries its own ground truth.
fn log_gpu_init_state(context: &str) {
    use std::path::Path;

    // /dev/nvidiactl appears at module load, /dev/nvidia<N> only after a
    // GPU's RmInitAdapter succeeds, /dev/nvidia-uvm with nvidia-uvm.
    let ctl = Path::new("/dev/nvidiactl").exists();
    let uvm = Path::new("/dev/nvidia-uvm").exists();
    let minors: Vec<String> = (0..32)
        .map(|n| format!("/dev/nvidia{n}"))
        .filter(|p| Path::new(p).exists())
        .collect();
    info!("gpu_init_state [{context}]: /dev/nvidiactl={ctl} /dev/nvidia-uvm={uvm} /dev/nvidia<N>={minors:?}");

    // Created at PCI probe, so non-empty even when every GPU later fails init.
    match std::fs::read_dir("/proc/driver/nvidia/gpus") {
        Ok(rd) => {
            for entry in rd.flatten() {
                let bdf = entry.file_name().to_string_lossy().into_owned();
                let minor = std::fs::read_to_string(entry.path().join("information"))
                    .ok()
                    .and_then(|s| {
                        s.lines()
                            .find(|l| l.contains("Device Minor"))
                            .map(str::to_owned)
                    })
                    .unwrap_or_else(|| "<no Device Minor>".to_owned());
                info!("gpu_init_state [{context}]: gpu {bdf}: {}", minor.trim());
            }
        }
        Err(e) => info!("gpu_init_state [{context}]: /proc/driver/nvidia/gpus unreadable: {e}"),
    }

    match std::fs::read_to_string("/proc/driver/nvidia/version") {
        Ok(s) => info!(
            "gpu_init_state [{context}]: driver: {}",
            s.lines().next().unwrap_or("<empty>")
        ),
        Err(e) => info!("gpu_init_state [{context}]: /proc/driver/nvidia/version unreadable: {e}"),
    }
}

/// VMs with GPU passthrough need driver setup, clock tuning,
/// and monitoring daemons before workloads can use the GPU.
/// On bare metal HGX systems (GPUs + NVSwitches), also starts
/// the fabric manager via the appropriate NVSwitch mode.
fn mode_gpu(init: &mut NVRC, nvswitch: Option<&str>, gpu_count: usize) {
    info!("mode_gpu: nvswitch={nvswitch:?} gpu_count={gpu_count}");
    modprobe::load("nvidia");
    modprobe::load("nvidia-uvm");
    log_gpu_init_state("post-modprobe");
    init.setup_uvm_tools();

    match nvswitch {
        Some("nvl4") => mode_nvl4(init, FABRIC_MODE_FULL, gpu_count),
        Some("nvl5") => mode_nvl5(init, FABRIC_MODE_FULL, gpu_count),
        // No NVSwitch or ConnectX fabric: on GB200 nvidia-imex brings up the
        // NVLink fabric instead (no-op if the image doesn't ship it).
        _ => init.nv_imex(),
    }

    init.nvidia_persistenced();

    init.nvidia_smi_lmc();
    init.nvidia_smi_lgc();
    init.nvidia_smi_pl();

    init.nv_hostengine();
    init.dcgm_exporter();
    nvidia_ctk_cdi();
    init.nvidia_smi_srs();
    init.health_checks();
}

/// NVSwitch NVL4 mode for HGX H100/H200/H800 systems (third-gen NVSwitch).
/// Service VM mode for NVLink 4.0 topologies in shared virtualization.
/// Starts fabric manager; the caller has already loaded the NVIDIA driver.
fn mode_nvl4(init: &mut NVRC, fabric_mode: u8, gpu_count: usize) {
    init.nv_fabricmanager(fabric_mode, "greedy", gpu_count);
    init.health_checks();
}

/// HGX Bx00 systems use CX7 bridges for NVLink management instead of direct GPU access.
/// GPUs are passed to tenant VMs; only the CX7 IB devices are visible here.
fn mode_nvl5(init: &mut NVRC, fabric_mode: u8, gpu_count: usize) {
    // ib_umad exposes /dev/umad* for InfiniBand MAD protocol access;
    // mlx5_ib creates /sys/class/infiniband/mlx5_* entries for the CX7 bridges.
    modprobe::load("ib_umad");
    modprobe::load("mlx5_ib");

    // CX7 port GUID identifies which bridge to use for fabric management
    init.port_guid = Some(
        infiniband::detect_port_guid()
            .expect("nvl5 requires SW_MNG IB device with valid port GUID"),
    );

    // NVLSM must initialize the NVLink subnet before FM can manage the fabric
    init.nv_nvlsm();
    init.health_checks();
    init.nv_fabricmanager(fabric_mode, "symmetric", gpu_count);
    init.health_checks();
}

fn main() {
    init::as_pid1();

    lockdown::set_panic_hook();
    let mut init = NVRC::default();
    mount::setup();
    net::loopback_up();
    kmsg::kernlog_setup();
    syslog::poll();
    // The UVM boots at the 1970 epoch; take host wall time from the KVM PTP
    // clock before the NVIDIA driver loads.
    time::sync_from_host();
    init.process_kernel_params(None);
    hash::self_exe();

    // Before disable_modules_loading() so dm-verity/erofs modules can still load.
    guest_extension_image::mount_all();

    // Expose gpu-extension libs/firmware before any driver load. No-op if absent.
    gpu_extension::setup();

    let detected = mode::detect();
    info!(
        "main: dispatching mode={} nvswitch={:?} gpu_count={}",
        detected.mode, detected.nvswitch, detected.gpu_count
    );
    match detected.mode {
        "cpu" => info!("executing cpu mode"),
        "gpu" => mode_gpu(&mut init, detected.nvswitch, detected.gpu_count),
        "servicevm-nvl4" => {
            modprobe::load("nvidia");
            mode_nvl4(&mut init, FABRIC_MODE_SHARED, detected.gpu_count);
        }
        "servicevm-nvl5" => mode_nvl5(&mut init, FABRIC_MODE_SHARED, detected.gpu_count),
        unknown => panic!("unknown mode: {unknown}"),
    }

    lockdown::disable_modules_loading();
    kata_agent::fork_agent(POLL_FOREVER);
}
