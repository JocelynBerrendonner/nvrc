// SPDX-License-Identifier: Apache-2.0
// Copyright (c) NVIDIA CORPORATION

mod config;
mod daemon;
mod execute;
mod infiniband;
mod kata_agent;
mod kernel_params;
mod kmsg;
mod lockdown;
mod macros;
mod mode;
mod modprobe;
mod mount;
mod net;
mod nvrc;
mod smi;
mod syslog;
mod time;
mod toolkit;

pub use macros::ResultExt;

#[cfg(test)]
mod test_utils;

#[macro_use]
extern crate log;
extern crate kernlog;

use daemon::FABRIC_MODE_FULL;
use daemon::FABRIC_MODE_SHARED;
use kata_agent::SYSLOG_POLL_FOREVER as POLL_FOREVER;
use nvrc::NVRC;
use toolkit::nvidia_ctk_cdi;

/// Diagnostic snapshot of every signal we could conceivably use to decide
/// whether the nvidia driver "initialized at least one GPU". Always
/// emitted at info level, even on success, because the data is tiny and
/// lets us cross-check on healthy hosts too.
fn log_gpu_init_state(context: &str) {
    info!("gpu_init_state [{context}]: collecting signals");

    // /dev/nvidiactl is created when the module loads; /dev/nvidia<N> is
    // created by nv_register_devices() AFTER at least one RmInitAdapter
    // succeeds; /dev/nvidia-uvm is created by the uvm module.
    let ctl = std::path::Path::new("/dev/nvidiactl").exists();
    let uvm = std::path::Path::new("/dev/nvidia-uvm").exists();
    let dev_nvidia_n: Vec<String> = (0..32)
        .filter(|n| std::path::Path::new(&format!("/dev/nvidia{n}")).exists())
        .map(|n| format!("/dev/nvidia{n}"))
        .collect();
    info!(
        "gpu_init_state [{context}]: /dev/nvidiactl={ctl} /dev/nvidia-uvm={uvm} \
         /dev/nvidia<N>={:?}",
        dev_nvidia_n
    );

    // /proc/driver/nvidia/gpus/<bdf>: created at PCI .probe() time, so it
    // can be non-empty even when every GPU later fails RmInitAdapter.
    match std::fs::read_dir("/proc/driver/nvidia/gpus") {
        Ok(rd) => {
            let bdfs: Vec<String> = rd
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            info!(
                "gpu_init_state [{context}]: /proc/driver/nvidia/gpus entries={:?}",
                bdfs
            );
        }
        Err(e) => info!(
            "gpu_init_state [{context}]: /proc/driver/nvidia/gpus unreadable: {e}"
        ),
    }

    // /proc/driver/nvidia/version exists once the module is loaded; the
    // contents tell us which driver build is running.
    match std::fs::read_to_string("/proc/driver/nvidia/version") {
        Ok(s) => info!(
            "gpu_init_state [{context}]: /proc/driver/nvidia/version first line: {}",
            s.lines().next().unwrap_or("<empty>")
        ),
        Err(e) => info!(
            "gpu_init_state [{context}]: /proc/driver/nvidia/version unreadable: {e}"
        ),
    }

    // Per-GPU registry: each BDF that completed probe gets an
    // `information` file with `Device Minor:` once it has a /dev minor
    // assigned. Worth dumping to cross-check against /dev/nvidia<N>.
    if let Ok(rd) = std::fs::read_dir("/proc/driver/nvidia/gpus") {
        for entry in rd.flatten() {
            let bdf = entry.file_name().to_string_lossy().into_owned();
            let info_path = entry.path().join("information");
            match std::fs::read_to_string(&info_path) {
                Ok(s) => {
                    let minor = s
                        .lines()
                        .find(|l| l.contains("Device Minor"))
                        .unwrap_or("<no Device Minor line>");
                    info!(
                        "gpu_init_state [{context}]: gpu {bdf} information: {}",
                        minor.trim()
                    );
                }
                Err(e) => info!(
                    "gpu_init_state [{context}]: gpu {bdf} information unreadable: {e}"
                ),
            }
        }
    }
}

/// VMs with GPU passthrough need driver setup, clock tuning,
/// and monitoring daemons before workloads can use the GPU.
/// On bare metal HGX systems (GPUs + NVSwitches), also starts
/// the fabric manager via the appropriate NVSwitch mode.
fn mode_gpu(init: &mut NVRC, nvswitch: Option<&str>, gpu_count: usize) {
    info!("mode_gpu: entering; nvswitch={:?}", nvswitch);
    modprobe::load("nvidia");
    modprobe::load("nvidia-uvm");

    // Snapshot every GPU-init signal at info level so we have ground
    // truth in the log for diagnosing a failing run on any platform.
    log_gpu_init_state("post-modprobe");

    match nvswitch {
        Some("nvl4") => mode_nvl4(init, FABRIC_MODE_FULL, gpu_count),
        Some("nvl5") => mode_nvl5(init, FABRIC_MODE_FULL, gpu_count),
        // No NVSwitch / SW_MNG fabric manager in this topology. On Blackwell
        // coherent-NVLink parts (GB200) the GPU fabric/clique is brought up by
        // nvidia-imex instead; start it here, after modprobe. nv_imex() no-ops
        // when the imex binary isn't shipped, so non-Blackwell / single-GPU
        // images are unaffected.
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
/// Loads NVIDIA driver and starts fabric manager. GPUs are assigned to service VM.
fn mode_nvl4(init: &mut NVRC, fabric_mode: u8, gpu_count: usize) {
    modprobe::load("nvidia");
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
    lockdown::set_panic_hook();
    let mut init = NVRC::default();
    mount::setup();
    net::loopback_up();
    kmsg::kernlog_setup();
    syslog::poll();
    // Pull host wall-time from the KVM PTP clock before loading the nvidia
    // driver: the UVM boots at the 1970 epoch, which trips the driver's
    // secTimerNs<osTimeNs assertions and breaks TLS/apt in the container.
    time::sync_from_host();
    init.process_kernel_params(None);

    let detected = mode::detect();
    info!(
        "main: dispatching mode={} nvswitch={:?}",
        detected.mode, detected.nvswitch
    );
    match detected.mode {
        "cpu" => info!("executing cpu mode"),
        "gpu" => mode_gpu(&mut init, detected.nvswitch, detected.gpu_count),
        "servicevm-nvl4" => mode_nvl4(&mut init, FABRIC_MODE_SHARED, detected.gpu_count),
        "servicevm-nvl5" => mode_nvl5(&mut init, FABRIC_MODE_SHARED, detected.gpu_count),
        unknown => panic!("unknown mode: {unknown}"),
    }

    lockdown::disable_modules_loading();
    kata_agent::fork_agent(POLL_FOREVER);
}
