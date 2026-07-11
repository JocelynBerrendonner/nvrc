// SPDX-License-Identifier: Apache-2.0
// Copyright (c) NVIDIA CORPORATION

//! Guest clock synchronization.
//!
//! The kata UVM boots with CLOCK_REALTIME at the Unix epoch (1970). Two things
//! break as a result:
//!   1. The nvidia driver floods `secTimerNs < osTimeNs` assertions
//!      (timer_gh100.c) because the OS clock sits far behind the monotonic
//!      hardware timer, and time-sensitive GPU / fabric init can misbehave.
//!   2. Anything inside the container that validates timestamps (TLS
//!      certificate validity, `apt update`) fails.
//!
//! Fix it by copying the host's wall-clock time into the guest via the KVM PTP
//! clock (/dev/ptp0). This needs no network and no external NTP server, and it
//! runs BEFORE `modprobe nvidia` so the driver sees a sane clock from the
//! start. The container shares this kernel, so its clock is fixed too.

use std::fs::File;
use std::os::unix::io::AsRawFd;

const PTP_DEV: &str = "/dev/ptp0";

/// POSIX dynamic clock id derived from an open PTP character-device fd:
/// `FD_TO_CLOCKID(fd) = ((~fd) << 3) | CLOCKFD`, with `CLOCKFD == 3`.
fn fd_to_clockid(fd: i32) -> libc::clockid_t {
    ((!fd) << 3) | 3
}

/// Set CLOCK_REALTIME from the KVM PTP clock (host wall time). Best-effort:
/// logs and returns without changing the clock if /dev/ptp0 is absent or the
/// PTP read fails, so a UVM without ptp_kvm still boots (it just keeps the
/// epoch clock, exactly as before this change).
pub fn sync_from_host() {
    let file = match File::open(PTP_DEV) {
        Ok(f) => f,
        Err(e) => {
            log::info!("time: {PTP_DEV} unavailable ({e}); leaving guest clock unchanged");
            return;
        }
    };

    let clockid = fd_to_clockid(file.as_raw_fd());
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };

    // SAFETY: `clockid` is derived from a live, open PTP device fd (`file` is
    // kept in scope for the duration of the call), and `ts` is a valid,
    // writable timespec out-parameter.
    if unsafe { libc::clock_gettime(clockid, &mut ts) } != 0 {
        log::warn!(
            "time: clock_gettime({PTP_DEV}) failed: {}; leaving guest clock unchanged",
            std::io::Error::last_os_error()
        );
        return;
    }

    // SAFETY: `ts` holds a valid time read from the PTP clock above.
    if unsafe { libc::clock_settime(libc::CLOCK_REALTIME, &ts) } != 0 {
        log::warn!(
            "time: clock_settime(CLOCK_REALTIME) failed: {}; leaving guest clock unchanged",
            std::io::Error::last_os_error()
        );
        return;
    }

    log::info!(
        "time: set CLOCK_REALTIME from {PTP_DEV} (host PTP) to {} epoch-seconds",
        ts.tv_sec
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fd_to_clockid_matches_posix_macro() {
        // FD_TO_CLOCKID(0) = (~0 << 3) | 3 = -8 | 3 = -5
        assert_eq!(fd_to_clockid(0), -5);
        // FD_TO_CLOCKID(3) = (~3 << 3) | 3 = (-4 << 3) | 3 = -32 | 3 = -29
        assert_eq!(fd_to_clockid(3), -29);
    }
}
