// SPDX-License-Identifier: Apache-2.0
// Copyright (c) NVIDIA CORPORATION

use crate::config::update_config_file;
use crate::execute::background;
use crate::kmsg;
use crate::macros::ResultExt;
use crate::nvrc::NVRC;
use std::fs;
use std::os::unix::fs::PermissionsExt;

/// UVM persistence mode keeps unified memory mappings alive between kernel launches,
/// avoiding expensive page migrations. Enabled by default for ML workloads.
fn persistenced_args(uvm_enabled: bool) -> Vec<&'static str> {
    if uvm_enabled {
        vec!["--verbose", "--uvm-persistence-mode"]
    } else {
        vec!["--verbose"]
    }
}

/// Hostengine needs a service account to avoid running as root, and /tmp as home
/// because the rootfs is read-only after init completes.
fn hostengine_args() -> &'static [&'static str] {
    &["--service-account", "nvidia-dcgm", "--home-dir", "/tmp"]
}

/// Kubernetes mode disables standalone HTTP server (we're behind kata-agent),
/// and we use the standard counters config shipped with the container image.
fn dcgm_exporter_args() -> &'static [&'static str] {
    &["-k", "-f", "/etc/dcgm-exporter/default-counters.csv"]
}

const FM_CONFIG: &str = "/usr/share/nvidia/nvswitch/fabricmanager.cfg";
const FM_RUNTIME_CONFIG: &str = "/run/fabricmanager.cfg";
const NVLSM_CONFIG: &str = "/usr/share/nvidia/nvlsm/nvlsm.conf";

/// FABRIC_MODE=0: full GPU passthrough, FM manages NVSwitches directly.
pub const FABRIC_MODE_FULL: u8 = 0;
/// FABRIC_MODE=1: shared NVSwitch virtualization, GPUs in tenant VMs.
pub const FABRIC_MODE_SHARED: u8 = 1;

/// Configurable path parameters allow testing with /bin/true instead of real
/// NVIDIA binaries that don't exist in the test environment.
impl NVRC {
    /// nvidia-persistenced keeps GPU state warm between container invocations,
    /// reducing cold-start latency. UVM persistence mode enables unified memory
    /// optimizations. Enabled by default since most workloads benefit from it.
    ///
    /// The readiness marker is `"nvidia-persistenced: Started ("` rather than
    /// the older `"Local RPC services initialized"`. Verified 2026-06-26 on
    /// Standard_ND96amsr_A100_v4 with NVIDIA driver 580.159.04:
    /// nvidia-persistenced --verbose emits, in order,
    ///     "Verbose syslog connection opened"
    ///     "Directory /var/run/nvidia-persistenced will not be removed on exit"
    ///     "Started (<PID>)"
    /// and then jumps straight to per-GPU `"device <BDF> - registered"` lines.
    /// The `"Local RPC services initialized"` string from older releases is
    /// never produced, so waiting for it deterministically panics after 120 s
    /// even on a successful run (this is the same regression class as the old
    /// fabricmanager `"FM starting NvLink Inband"` marker -- see
    /// `nv_fabricmanager` above). `"Started ("` is emitted immediately after
    /// the daemon completes startup and enters its main accept loop, which is
    /// the actual point we want to gate downstream nvidia-smi calls on.
    pub fn nvidia_persistenced(&mut self) {
        let mut reader = kmsg::open_kmsg("/dev/kmsg");
        self.spawn_persistenced("/var/run/nvidia-persistenced", "/bin/nvidia-persistenced");
        kmsg::wait_for_marker(&mut reader, "nvidia-persistenced: Started (", 120);
    }

    fn spawn_persistenced(&mut self, run_dir: &str, bin: &str) {
        fs::create_dir_all(run_dir).or_panic(format_args!("create_dir_all {run_dir}"));
        let uvm_enabled = self.uvm_persistence_mode.unwrap_or(true);
        let args = persistenced_args(uvm_enabled);
        let child = background(bin, &args);
        self.track_daemon("nvidia-persistenced", child);
    }

    /// nv-hostengine is the DCGM backend daemon. Only started when DCGM monitoring
    /// is explicitly requested - not needed for basic GPU workloads.
    pub fn nv_hostengine(&mut self) {
        self.spawn_hostengine("/bin/nv-hostengine")
    }

    fn spawn_hostengine(&mut self, bin: &str) {
        if !self.dcgm_enabled.unwrap_or(false) {
            return;
        }
        let child = background(bin, hostengine_args());
        self.track_daemon("nv-hostengine", child);
    }

    /// dcgm-exporter exposes GPU metrics for Prometheus. Only started when DCGM
    /// is enabled - adds overhead so disabled by default.
    pub fn dcgm_exporter(&mut self) {
        self.spawn_dcgm_exporter("/bin/dcgm-exporter")
    }

    fn spawn_dcgm_exporter(&mut self, bin: &str) {
        if !self.dcgm_enabled.unwrap_or(false) {
            return;
        }
        let child = background(bin, dcgm_exporter_args());
        self.track_daemon("dcgm-exporter", child);
    }

    /// NVSwitch fabric manager is only needed for multi-GPU NVLink topologies.
    /// Disabled by default since most VMs have single GPUs.
    ///
    /// `gpu_count` is the number of GPUs detected during topology detection;
    /// it's passed in (rather than re-counted here) so the same source of
    /// truth that drove mode dispatch also drives the kmsg wait.
    pub fn nv_fabricmanager(&mut self, fabric_mode: u8, rail_policy: &str, gpu_count: usize) {
        fs::copy(FM_CONFIG, FM_RUNTIME_CONFIG)
            .or_panic(format_args!("copy {FM_CONFIG} to {FM_RUNTIME_CONFIG}"));
        self.configure_fabricmanager(FM_RUNTIME_CONFIG, fabric_mode, rail_policy);
        fs::set_permissions(FM_RUNTIME_CONFIG, fs::Permissions::from_mode(0o400))
            .or_panic(format_args!("set permissions {FM_RUNTIME_CONFIG}"));
        self.spawn_fabricmanager("/bin/nv-fabricmanager");

        // Driver-emitted kernel marker (one per GPU) instead of the
        // userspace fabricmanager marker `"FM starting NvLink Inband"`.
        //
        // Background: the fabricmanager binary shipped with NVIDIA driver
        // 580.* does NOT emit the FM marker even with LOG_USE_SYSLOG=0 +
        // DAEMONIZE=0 + LOG_FILE_NAME=/dev/stderr + LOG_LEVEL=5. Verified
        // 2026-06-25 on Standard_ND96amsr_A100_v4 (HGX A100 8-GPU + 6
        // NVSwitch): fabricmanager runs to completion (nvidia-smi topo -m
        // shows NV12 between every GPU pair = fully trained NVLink fabric),
        // but its stderr stays empty and nothing matching
        // `FM starting NvLink Inband` ever reaches kmsg or /run/syslog.log.
        //
        // The in-kernel nvidia.ko driver, however, reliably emits
        // `NVRM: knvlinkSetUniqueFabricBaseAddress_GV100: Fabric base
        // addr X is assigned to GPU N` once per GPU as each GPU is
        // registered into the trained fabric. That sequence completes
        // when (and only when) fabricmanager has finished bring-up.
        // Waiting for `gpu_count` matches in /dev/kmsg gives us a
        // driver-version-stable "fabric is up" signal that doesn't
        // depend on fabricmanager's userspace logging behavior.
        //
        // Marker is `"NVRM: knvlinkSetUniqueFabricBaseAddress_GV100"`
        // -- WITH the `"NVRM: "` prefix -- to disambiguate the real
        // driver printk from any other line that happens to contain
        // the function name. The driver always emits this string with
        // the `NVRM: ` prefix; NVRC's own kernlog-routed info!() lines
        // do not. Necessary because previously (`knvlink...` substring
        // only) the waiter self-fed off lines that NVRC itself wrote
        // about the marker. Verified 2026-06-26: with the bare
        // substring marker, all 8 matches completed in 1.24 ms because
        // NVRC's own startup banner `"waiting for 8 occurrence(s) of
        // 'knvlink...'"` matched first, then 7 of its own `"matched
        // X/8: knvlink..."` lines matched themselves -- nvidia-smi
        // inside the container then hung because the fabric was still
        // training. kmsg.rs::wait_for_kmsg_count was also scrubbed to
        // not echo the marker text in its own info!() lines.
        //
        // 300 s timeout (vs. the old 120 s) leaves headroom for benches
        // with emulated MMIO (L1VH, nested virt) where each NVSwitch
        // BAR write is significantly slower than bare metal.
        kmsg::wait_for_kmsg_count(
            "NVRM: knvlinkSetUniqueFabricBaseAddress_GV100",
            gpu_count,
            300,
        );
    }

    fn spawn_fabricmanager(&mut self, bin: &str) {
        let mut args = vec!["-c", FM_RUNTIME_CONFIG];
        let guid_owned: String;
        if let Some(ref guid) = self.port_guid {
            guid_owned = guid.clone();
            args.push("-g");
            args.push(&guid_owned);
        }
        let child = background(bin, &args);
        self.track_daemon("nv-fabricmanager", child);
    }

    /// CX7 bridges require NVLSM to manage NVLink subnet before FM can initialize the fabric.
    pub fn nv_nvlsm(&mut self) {
        self.spawn_nvlsm("/sbin/nvlsm")
    }

    fn spawn_nvlsm(&mut self, bin: &str) {
        let Some(ref guid) = self.port_guid else {
            return;
        };
        let guid_owned = guid.clone();
        let args = vec!["-F", NVLSM_CONFIG, "-g", &guid_owned, "-f", "stdout"];
        let child = background(bin, &args);
        self.track_daemon("nvlsm", child);
    }

    /// Write FABRIC_MODE and PARTITION_RAIL_POLICY to fabricmanager.cfg.
    /// FABRIC_MODE: 0 = bare metal (GPUs local), 1 = service VM (GPUs in tenant VMs)
    /// PARTITION_RAIL_POLICY: "greedy" (NVL4) or "symmetric" (NVL5, required for CC on Blackwell)
    ///
    /// PARTITION_RAIL_POLICY is only emitted when `fabric_mode != FABRIC_MODE_FULL`:
    /// fabricmanager in `FABRIC_MODE=0` rejects the key with
    /// `unsupported config item PARTITION_RAIL_POLICY is specified in fabric
    /// manager config file` (verified 2026-06-26 on Standard_ND96amsr_A100_v4,
    /// FM 580.159.04). The warning is benign — FM proceeds normally — but it
    /// pollutes the openvmm-guest journal and falsely implicates fabricmanager
    /// during triage. The shared-NVSwitch path (`FABRIC_MODE=1`) is where the
    /// rail-policy knob actually applies, so we keep emitting it there.
    ///
    /// Also force four settings that make fabricmanager observable and reapable
    /// from inside NVRC's UVM (regardless of what the shipped default cfg says):
    ///
    /// * `LOG_FILE_NAME=/dev/stderr` — fabricmanager logs to stderr instead of
    ///   `/var/log/fabricmanager.log`. `background()` wires stderr to /dev/kmsg,
    ///   which the kernel forwards to the hvc0 console, which the host captures
    ///   as the `openvmm-guest:` journal stream. The default file destination
    ///   lives on the UVM's tmpfs and is unrecoverable once NVRC panics and
    ///   the UVM dies, so debugging fabricmanager hangs (e.g. ENOENT on
    ///   `/usr/bin/nvidia-modprobe`, missing topology files, NvLink training
    ///   failures) is impossible without this.
    ///
    ///   IMPORTANT: must be `/dev/stderr` (a real fd, == /proc/self/fd/2), NOT
    ///   empty. Verified 2026-06-23 on Standard_ND96amsr_A100_v4: fabricmanager
    ///   treats `LOG_FILE_NAME=` (empty) as "fall back to compile-time default
    ///   = /var/log/fabricmanager.log", which silently traps every log line on
    ///   the UVM tmpfs. With `=/dev/stderr` the output flows to the console.
    /// * `LOG_LEVEL=5` — DEBUG. Default is 4 = INFO which only emits the
    ///   `FM starting NvLink Inband` marker and fatal errors. DEBUG gives us
    ///   the per-fabric-init-step output that's actually useful when
    ///   diagnosing "fabricmanager spawned and went silent" hangs.
    /// * `LOG_USE_SYSLOG=0` — the chiseled UVM has no syslog daemon. With
    ///   `=1` (the shipped default) fabricmanager prefers syslog and silently
    ///   drops lines whose syslog write fails, leaving us with even fewer
    ///   diagnostic breadcrumbs than the file destination above.
    /// * `DAEMONIZE=0` — correctness fix: with `=1` (the shipped default)
    ///   fabricmanager forks and the parent exits immediately, so NVRC's
    ///   `track_daemon("nv-fabricmanager", child)` ends up tracking a PID
    ///   that's already gone, the real fabricmanager becomes a re-parented
    ///   orphan that NVRC can't reap, and the stderr fd that `background()`
    ///   wired up disconnects on the parent exit. `=0` keeps fabricmanager
    ///   in the foreground so NVRC owns the right PID and stderr stays
    ///   attached for the whole process lifetime.
    fn configure_fabricmanager(&self, cfg_path: &str, fabric_mode: u8, rail_policy: &str) {
        let fm = &fabric_mode.to_string();
        let mut updates: Vec<(&str, &str)> = vec![
            ("FABRIC_MODE", fm.as_str()),
            ("LOG_FILE_NAME", "/dev/stderr"),
            ("LOG_LEVEL", "5"),
            ("LOG_USE_SYSLOG", "0"),
            ("DAEMONIZE", "0"),
        ];
        // PARTITION_RAIL_POLICY is only honored by fabricmanager in shared-
        // NVSwitch mode (FABRIC_MODE=1). In FABRIC_MODE=0 (GPUs local) FM
        // logs an "unsupported config item" warning and ignores the value.
        if fabric_mode != FABRIC_MODE_FULL {
            updates.push(("PARTITION_RAIL_POLICY", rail_policy));
        }
        update_config_file(cfg_path, &updates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // === Args builder tests ===

    #[test]
    fn test_persistenced_args_with_uvm() {
        let args = persistenced_args(true);
        assert_eq!(args, vec!["--verbose", "--uvm-persistence-mode"]);
    }

    #[test]
    fn test_persistenced_args_without_uvm() {
        let args = persistenced_args(false);
        assert_eq!(args, vec!["--verbose"]);
    }

    #[test]
    fn test_hostengine_args() {
        let args = hostengine_args();
        assert_eq!(
            args,
            &["--service-account", "nvidia-dcgm", "--home-dir", "/tmp"]
        );
    }

    #[test]
    fn test_dcgm_exporter_args() {
        let args = dcgm_exporter_args();
        assert_eq!(
            args,
            &["-k", "-f", "/etc/dcgm-exporter/default-counters.csv"]
        );
    }

    // === Skip path tests ===

    #[test]
    fn test_nv_hostengine_skipped_by_default() {
        // DCGM disabled by default - should be a no-op, no daemon spawned
        let mut nvrc = NVRC::default();
        nvrc.nv_hostengine();
        nvrc.health_checks();
    }

    #[test]
    fn test_dcgm_exporter_skipped_by_default() {
        let mut nvrc = NVRC::default();
        nvrc.dcgm_exporter();
    }

    #[test]
    fn test_nv_fabricmanager_gpu_mode() {
        use tempfile::NamedTempFile;

        let tmpfile = NamedTempFile::new().unwrap();
        let cfg = tmpfile.path().to_str().unwrap();
        fs::write(cfg, "FABRIC_MODE=0\n").unwrap();

        let mut nvrc = NVRC::default();
        nvrc.configure_fabricmanager(cfg, FABRIC_MODE_FULL, "greedy");
        nvrc.spawn_fabricmanager("/bin/true");

        let content = fs::read_to_string(cfg).unwrap();
        assert!(content.contains("FABRIC_MODE=0"));
        nvrc.health_checks();
    }

    #[test]
    fn test_spawn_persistenced_success() {
        let tmpdir = TempDir::new().unwrap();
        let run_dir = tmpdir.path().join("nvidia-persistenced");

        let mut nvrc = NVRC::default();
        nvrc.spawn_persistenced(run_dir.to_str().unwrap(), "/bin/true");

        // Directory should be created
        assert!(run_dir.exists());

        // Daemon should be tracked and exit cleanly
        nvrc.health_checks();
    }

    #[test]
    fn test_spawn_persistenced_uvm_disabled() {
        let tmpdir = TempDir::new().unwrap();
        let run_dir = tmpdir.path().join("nvidia-persistenced");

        let mut nvrc = NVRC::default();
        nvrc.uvm_persistence_mode = Some(false); // Tests the else branch for args
        nvrc.spawn_persistenced(run_dir.to_str().unwrap(), "/bin/true");
    }

    #[test]
    fn test_spawn_hostengine_success() {
        let mut nvrc = NVRC::default();
        nvrc.dcgm_enabled = Some(true);
        nvrc.spawn_hostengine("/bin/true");
        nvrc.health_checks();
    }

    #[test]
    fn test_spawn_dcgm_exporter_success() {
        let mut nvrc = NVRC::default();
        nvrc.dcgm_enabled = Some(true);
        nvrc.spawn_dcgm_exporter("/bin/true");
    }

    #[test]
    fn test_spawn_fabricmanager_success() {
        let mut nvrc = NVRC::default();
        nvrc.spawn_fabricmanager("/bin/true");
    }

    #[test]
    fn test_spawn_fabricmanager_with_port_guid() {
        let mut nvrc = NVRC::default();
        nvrc.port_guid = Some("0xdeadbeef".to_string());
        nvrc.spawn_fabricmanager("/bin/true");
        nvrc.health_checks();
    }

    #[test]
    fn test_spawn_nvlsm_success() {
        let mut nvrc = NVRC::default();
        nvrc.port_guid = Some("0xdeadbeef".to_string());
        nvrc.spawn_nvlsm("/bin/true");
        nvrc.health_checks();
    }

    #[test]
    fn test_spawn_nvlsm_skipped_without_guid() {
        let mut nvrc = NVRC::default();
        // port_guid is None, should be a no-op
        nvrc.spawn_nvlsm("/bin/true");
    }

    #[test]
    fn test_spawn_persistenced_binary_not_found() {
        use std::panic;

        let tmpdir = TempDir::new().unwrap();
        let run_dir = tmpdir.path().join("nvidia-persistenced");

        let result = panic::catch_unwind(|| {
            let mut nvrc = NVRC::default();
            nvrc.spawn_persistenced(run_dir.to_str().unwrap(), "/nonexistent/binary");
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_health_checks_empty() {
        let mut nvrc = NVRC::default();
        nvrc.health_checks();
    }

    // === Fabricmanager configuration tests ===

    #[test]
    fn test_configure_fabricmanager_bare_metal() {
        use tempfile::NamedTempFile;

        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        fs::write(path, "").unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_FULL, "greedy");

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("FABRIC_MODE=0"));
    }

    #[test]
    fn test_configure_fabricmanager_servicevm_nvl4() {
        use tempfile::NamedTempFile;

        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        fs::write(path, "").unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_SHARED, "greedy");

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("FABRIC_MODE=1"));
    }

    #[test]
    fn test_configure_fabricmanager_servicevm_nvl5() {
        use tempfile::NamedTempFile;

        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        fs::write(path, "").unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_SHARED, "symmetric");

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("FABRIC_MODE=1"));
    }

    #[test]
    fn test_configure_fabricmanager_updates_existing() {
        use tempfile::NamedTempFile;

        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        fs::write(path, "FABRIC_MODE=0\n").unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_SHARED, "greedy");

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("FABRIC_MODE=1"));
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.starts_with("FABRIC_MODE="))
                .count(),
            1
        );
    }

    #[test]
    fn test_configure_fabricmanager_preserves_other_config() {
        use tempfile::NamedTempFile;

        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        fs::write(path, "# Comment\nOTHER_SETTING=value\nFABRIC_MODE=0\n").unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_SHARED, "greedy");

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("# Comment"));
        assert!(content.contains("OTHER_SETTING=value"));
        assert!(content.contains("FABRIC_MODE=1"));
    }

    #[test]
    fn test_configure_fabricmanager_nvl4_greedy_rail_policy() {
        use tempfile::NamedTempFile;

        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        fs::write(path, "").unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_SHARED, "greedy");

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("PARTITION_RAIL_POLICY=greedy"));
    }

    #[test]
    fn test_configure_fabricmanager_nvl5_symmetric_rail_policy() {
        use tempfile::NamedTempFile;

        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        fs::write(path, "").unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_SHARED, "symmetric");

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("PARTITION_RAIL_POLICY=symmetric"));
    }

    #[test]
    fn test_configure_fabricmanager_full_mode_omits_rail_policy() {
        use tempfile::NamedTempFile;

        // FABRIC_MODE=0 (GPU passthrough / bare metal) rejects
        // PARTITION_RAIL_POLICY with an "unsupported config item" warning,
        // so configure_fabricmanager must not emit it. Holds regardless of
        // the rail_policy argument NVRC was called with (mode_nvl5 still
        // passes "symmetric" even when we're in FABRIC_MODE_FULL).
        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        fs::write(path, "").unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_FULL, "symmetric");

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("FABRIC_MODE=0"));
        assert!(
            !content.contains("PARTITION_RAIL_POLICY"),
            "PARTITION_RAIL_POLICY must not be written in FABRIC_MODE_FULL; got:\n{}",
            content
        );
    }

    #[test]
    fn test_configure_fabricmanager_full_mode_preserves_existing_rail_policy() {
        use tempfile::NamedTempFile;

        // If the shipped cfg already contains PARTITION_RAIL_POLICY (it
        // doesn't today, but defensively): configure_fabricmanager does NOT
        // remove existing keys, it only updates the set it knows about. So
        // any pre-existing line is left as-is; we just don't add one of our
        // own. Document this contract in a test.
        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        fs::write(path, "PARTITION_RAIL_POLICY=greedy\n").unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_FULL, "symmetric");

        let content = fs::read_to_string(path).unwrap();
        // Existing line is preserved (untouched).
        assert!(content.contains("PARTITION_RAIL_POLICY=greedy"));
        // And we did NOT add a second line with the new value.
        assert!(!content.contains("PARTITION_RAIL_POLICY=symmetric"));
    }

    #[test]
    fn test_configure_fabricmanager_all_settings() {
        use tempfile::NamedTempFile;

        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        fs::write(path, "").unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_SHARED, "symmetric");

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("FABRIC_MODE=1"));
        assert!(content.contains("PARTITION_RAIL_POLICY=symmetric"));
    }

    /// configure_fabricmanager() must unconditionally emit four diagnostic /
    /// correctness overrides on top of whatever the shipped fabricmanager.cfg
    /// default says, regardless of fabric_mode or rail_policy:
    ///   * LOG_FILE_NAME=/dev/stderr  (logs to stderr -> /dev/kmsg -> console)
    ///   * LOG_LEVEL=5                (DEBUG -- captures per-init-step output)
    ///   * LOG_USE_SYSLOG=0           (no syslogd in the chiseled UVM)
    ///   * DAEMONIZE=0                (keep PID stable for NVRC's track_daemon)
    /// Without these, fabricmanager hangs/errors are invisible from the host
    /// journal and NVRC tracks a transient PID instead of the real worker.
    /// `/dev/stderr` is required because empty `LOG_FILE_NAME=` is interpreted
    /// by fabricmanager as "fall back to /var/log/fabricmanager.log".
    #[test]
    fn test_configure_fabricmanager_emits_diagnostic_overrides() {
        use tempfile::NamedTempFile;

        let tmpfile = NamedTempFile::new().unwrap();
        let path = tmpfile.path().to_str().unwrap();
        // Start from a cfg that has the shipped defaults so we exercise the
        // "update existing key in place" code path, not the "append" path.
        fs::write(
            path,
            "LOG_FILE_NAME=/var/log/fabricmanager.log\n\
             LOG_LEVEL=4\n\
             LOG_USE_SYSLOG=1\n\
             DAEMONIZE=1\n",
        )
        .unwrap();

        let nvrc = NVRC::default();
        nvrc.configure_fabricmanager(path, FABRIC_MODE_FULL, "greedy");

        let content = fs::read_to_string(path).unwrap();
        let has_line = |k: &str| content.lines().any(|l| l.trim() == k);
        // Exact-line matches catch the stale defaults being left behind.
        assert!(has_line("LOG_FILE_NAME=/dev/stderr"), "got:\n{}", content);
        assert!(has_line("LOG_LEVEL=5"), "got:\n{}", content);
        assert!(has_line("LOG_USE_SYSLOG=0"), "got:\n{}", content);
        assert!(has_line("DAEMONIZE=0"), "got:\n{}", content);
        // And ensure the original defaults are GONE (not just shadowed).
        assert!(
            !content.contains("LOG_FILE_NAME=/var/log/fabricmanager.log"),
            "stale LOG_FILE_NAME survived:\n{}",
            content
        );
        assert!(!has_line("LOG_LEVEL=4"), "stale LOG_LEVEL survived:\n{}", content);
        assert!(!has_line("LOG_USE_SYSLOG=1"), "stale LOG_USE_SYSLOG survived:\n{}", content);
        assert!(!has_line("DAEMONIZE=1"), "stale DAEMONIZE survived:\n{}", content);
    }
}
