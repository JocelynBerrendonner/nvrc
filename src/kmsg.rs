// SPDX-License-Identifier: Apache-2.0
// Copyright (c) NVIDIA CORPORATION

use crate::macros::ResultExt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader};
use std::os::unix::fs::OpenOptionsExt;
use std::sync::Once;
use std::time::{Duration, Instant};

static KERNLOG_INIT: Once = Once::new();

/// Socket buffer size (16MB = 16 * 1024 * 1024 = 16777216 bytes).
/// Large buffers prevent message loss during high-throughput GPU operations
/// where NVIDIA drivers may emit bursts of diagnostic data.
const SOCKET_BUFFER_SIZE: &str = "16777216";

/// Initialize kernel logging and tune socket buffer sizes.
/// Large buffers (16MB) prevent message loss during high-throughput GPU operations
/// where drivers may emit bursts of diagnostic data.
pub fn kernlog_setup() {
    KERNLOG_INIT.call_once(|| {
        let _ = kernlog::init();
    });
    log::set_max_level(log::LevelFilter::Off);
    for path in [
        "/proc/sys/net/core/rmem_default",
        "/proc/sys/net/core/wmem_default",
        "/proc/sys/net/core/rmem_max",
        "/proc/sys/net/core/wmem_max",
    ] {
        fs::write(path, SOCKET_BUFFER_SIZE.as_bytes()).or_panic(format_args!("write {path}"));
    }
}

/// Get a file handle for kernel message output.
/// Routes to /dev/kmsg when debug logging is enabled for visibility in dmesg,
/// otherwise /dev/null to suppress noise in production.
pub fn kmsg() -> File {
    kmsg_at(if log_enabled!(log::Level::Debug) {
        "/dev/kmsg"
    } else {
        "/dev/null"
    })
}

/// Open syslog file for reading daemon startup markers.
/// Maps /dev/kmsg to /run/syslog.log because daemon synchronization needs to
/// work without trace logging enabled. File-based sync is simpler and more
/// reliable than trying to coordinate log levels between writer and reader.
pub fn open_kmsg(path: &str) -> BufReader<File> {
    let log_path = if path == "/dev/kmsg" {
        crate::syslog::SYSLOG_FILE_PATH
    } else {
        path
    };

    // Try read-only first; if missing, create with secure perms then reopen
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(log_path)
        .or_else(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                // Create with restrictive permissions
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(log_path)?;
                // Reopen read-only
                OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_NONBLOCK)
                    .open(log_path)
            } else {
                Err(e)
            }
        })
        .or_panic(format_args!("open {log_path}"));

    BufReader::new(file)
}

/// Block until `marker` appears in `reader` or `timeout_secs` expires.
/// Calls try_poll() to drain /dev/log socket and write messages to file that
/// we're reading from. This loop is the syslog daemon for our minimal init.
pub fn wait_for_marker(reader: &mut BufReader<File>, marker: &str, timeout_secs: u32) {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs as u64);
    let mut line = String::new();

    loop {
        crate::syslog::try_poll();
        if Instant::now() > deadline {
            panic!("timeout waiting for: {marker}");
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => std::thread::sleep(Duration::from_millis(500)),
            Ok(_) if line.contains(marker) => {
                info!("{marker}");
                return;
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(500)),
        }
    }
}

/// Block until `marker` substring appears `count` times in `/dev/kmsg`,
/// or `timeout_secs` expires.
///
/// Unlike [`wait_for_marker`], this reads `/dev/kmsg` DIRECTLY (not via
/// the syslog-mapped file path) because the markers of interest are
/// kernel printks emitted by drivers like `nvidia.ko`, which never reach
/// `/dev/log` and therefore never reach `/run/syslog.log`.
///
/// Only counts messages emitted AFTER this function is called: we
/// `lseek(SEEK_END)` past the existing kmsg ring buffer before reading
/// so historical kernel messages (from earlier boot, module load, etc.)
/// don't accidentally satisfy the wait.
///
/// Designed for the NVRC fabricmanager synchronization use-case: the
/// fabricmanager binary shipped with NVIDIA driver 580.* does NOT emit
/// the `FM starting NvLink Inband` marker string that earlier versions
/// did. Instead, the in-kernel NVIDIA driver emits
/// `NVRM: knvlinkSetUniqueFabricBaseAddress_GV100: Fabric base addr X
/// is assigned to GPU N` once per GPU as the driver registers each
/// GPU into the trained NVLink fabric. Waiting for `gpu_count` copies
/// of `knvlinkSetUniqueFabricBaseAddress_GV100` is a driver-version-
/// stable equivalent of the old fabricmanager marker.
pub fn wait_for_kmsg_count(marker: &str, count: usize, timeout_secs: u32) {
    if count == 0 {
        // Log the count/timeout (caller-supplied integers), NOT the marker
        // text -- see the long comment in the loop below for why mentioning
        // `marker` in any info!() that the kernlog crate writes to /dev/kmsg
        // creates a self-feedback loop.
        info!("wait_for_kmsg_count: count=0, returning immediately");
        return;
    }

    // Open /dev/kmsg with O_NONBLOCK so reads return EAGAIN/WouldBlock
    // when no new messages are pending, letting us sleep and re-check
    // the deadline.
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open("/dev/kmsg")
        .or_panic(format_args!("open /dev/kmsg"));

    // /dev/kmsg supports seek: SEEK_END moves the read position past
    // the last buffered message, so subsequent reads return only new
    // messages emitted AFTER this point.
    use std::io::Seek;
    let mut file = file;
    let _ = file.seek(std::io::SeekFrom::End(0));
    let mut reader = BufReader::new(file);

    let deadline = Instant::now() + Duration::from_secs(timeout_secs as u64);
    let mut line = String::new();
    let mut seen: usize = 0;

    // CAREFUL: do NOT include `{marker}` in any info!() inside this function.
    // The NVRC build links the `kernlog` crate as its log backend, which
    // routes every info!() through /dev/kmsg. Any line we emit then becomes
    // the next line we read back from /dev/kmsg. If that line contains the
    // marker substring (because we said "waiting for 'knvlink...'" or
    // "matched 1/8: knvlink..."), we count our OWN log line as a match,
    // log a new "matched 2/8" line that also contains the substring, count
    // that, and so on -- the waiter completes in ~1 ms instead of waiting
    // for the actual driver-emitted kernel printk. Verified 2026-06-26 on
    // Standard_ND96amsr_A100_v4: 8 GPUs supposedly registered into the
    // fabric in 1.24 ms, then nvidia-smi inside the container hung because
    // the fabric was still training. Keep all status logging in this
    // function marker-free.
    info!("wait_for_kmsg_count: waiting for {count} occurrence(s) in /dev/kmsg (timeout {timeout_secs}s)");

    loop {
        if Instant::now() > deadline {
            panic!("timeout waiting for {marker} ({seen}/{count} seen)");
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => std::thread::sleep(Duration::from_millis(200)),
            Ok(_) if line.contains(marker) => {
                seen += 1;
                info!("wait_for_kmsg_count: matched {seen}/{count}");
                if seen >= count {
                    return;
                }
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(200)),
        }
    }
}

/// Internal: open the given path for writing. Extracted for testability.
fn kmsg_at(path: &str) -> File {
    OpenOptions::new()
        .write(true)
        .open(path)
        .or_panic(format_args!("open {path}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::require_root;
    use serial_test::serial;
    use std::io::Write;
    use std::panic;
    use tempfile::NamedTempFile;

    #[test]
    fn test_kmsg_at_dev_null() {
        // /dev/null is always writable, no root needed
        let _file = kmsg_at("/dev/null");
    }

    #[test]
    fn test_kmsg_at_nonexistent() {
        let result = panic::catch_unwind(|| {
            kmsg_at("/nonexistent/path");
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_kmsg_at_temp_file() {
        // Create a temp file to verify we can write to it
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_str().unwrap();
        let mut file = kmsg_at(path);
        assert!(file.write_all(b"test").is_ok());
    }

    #[test]
    #[serial]
    fn test_kmsg_routes_to_dev_null_when_log_off() {
        // Default log level is Off, so kmsg() should open /dev/null
        log::set_max_level(log::LevelFilter::Off);
        let _file = kmsg();
    }

    #[test]
    #[serial]
    fn test_kmsg_routes_to_kmsg_when_debug() {
        require_root();
        // When debug is enabled, kmsg() should open /dev/kmsg
        log::set_max_level(log::LevelFilter::Debug);
        let _file = kmsg();
        log::set_max_level(log::LevelFilter::Off);
    }

    #[test]
    #[serial]
    fn test_kernlog_setup() {
        require_root();

        const PATHS: [&str; 4] = [
            "/proc/sys/net/core/rmem_default",
            "/proc/sys/net/core/wmem_default",
            "/proc/sys/net/core/rmem_max",
            "/proc/sys/net/core/wmem_max",
        ];

        // RAII guard to restore original values after test
        struct Restore(Vec<(&'static str, String)>);
        impl Drop for Restore {
            fn drop(&mut self) {
                for (path, value) in &self.0 {
                    let _ = fs::write(path, value.as_bytes());
                }
            }
        }

        let saved: Vec<_> = PATHS
            .iter()
            .filter_map(|&p| fs::read_to_string(p).ok().map(|v| (p, v)))
            .collect();
        let _restore = Restore(saved);

        kernlog_setup();

        for &path in &PATHS {
            let v = fs::read_to_string(path).expect("should read sysctl");
            assert_eq!(
                v.trim(),
                SOCKET_BUFFER_SIZE,
                "sysctl {} should be {}",
                path,
                SOCKET_BUFFER_SIZE
            );
        }
    }

    // === wait_for_marker tests ===

    #[test]
    fn test_wait_for_marker_finds_marker() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "some noise").unwrap();
        writeln!(tmp, "FM starting NvLink Inband foo").unwrap();
        writeln!(tmp, "more noise").unwrap();
        tmp.flush().unwrap();

        wait_for_marker(
            &mut open_kmsg(tmp.path().to_str().unwrap()),
            "FM starting NvLink Inband",
            5,
        );
    }

    #[test]
    fn test_wait_for_marker_finds_marker_at_end() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "line 1").unwrap();
        writeln!(tmp, "line 2").unwrap();
        writeln!(tmp, "FM starting NvLink Inband").unwrap();
        tmp.flush().unwrap();

        wait_for_marker(
            &mut open_kmsg(tmp.path().to_str().unwrap()),
            "FM starting NvLink Inband",
            5,
        );
    }

    #[test]
    fn test_wait_for_marker_no_marker_panics() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "no match here").unwrap();
        tmp.flush().unwrap();

        let result = panic::catch_unwind(|| {
            wait_for_marker(
                &mut open_kmsg(tmp.path().to_str().unwrap()),
                "FM starting NvLink Inband",
                1,
            );
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_wait_for_marker_empty_file_panics() {
        let tmp = NamedTempFile::new().unwrap();

        let result = panic::catch_unwind(|| {
            wait_for_marker(
                &mut open_kmsg(tmp.path().to_str().unwrap()),
                "FM starting NvLink Inband",
                1,
            );
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_wait_for_marker_nonexistent_file_panics() {
        let result = panic::catch_unwind(|| {
            wait_for_marker(&mut open_kmsg("/nonexistent/path"), "marker", 1);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_wait_for_marker_partial_match_not_enough() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "FM starting").unwrap();
        writeln!(tmp, "NvLink Inband").unwrap();
        tmp.flush().unwrap();

        // Marker spans two lines — should not match
        let result = panic::catch_unwind(|| {
            wait_for_marker(
                &mut open_kmsg(tmp.path().to_str().unwrap()),
                "FM starting NvLink Inband",
                1,
            );
        });
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_wait_for_marker_on_dev_kmsg() {
        require_root();

        // Clear any previous test data to avoid false positives
        let _ = fs::remove_file(crate::syslog::SYSLOG_FILE_PATH);

        // With always-file architecture, open_kmsg("/dev/kmsg") always reads from syslog file
        let mut reader = open_kmsg("/dev/kmsg");
        let marker = "NVRC_TEST_MARKER_12345";

        // Write directly to the syslog file (simulating what syslog.rs does)
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(crate::syslog::SYSLOG_FILE_PATH)
            .expect("open syslog file");
        writeln!(file, "{}", marker).expect("write marker");
        file.flush().expect("flush");

        wait_for_marker(&mut reader, marker, 5);
    }
}
