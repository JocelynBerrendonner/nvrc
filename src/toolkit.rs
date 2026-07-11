// SPDX-License-Identifier: Apache-2.0
// Copyright (c) NVIDIA CORPORATION

//! NVIDIA Container Toolkit (nvidia-ctk) integration.
//!
//! Generates CDI (Container Device Interface) specs so container runtimes
//! can discover and mount GPU devices without needing the legacy hook.

use crate::execute::foreground;
use std::fs;
use std::path::Path;

const NVIDIA_CTK: &str = "/bin/nvidia-ctk";
const CDI_SPEC: &str = "/var/run/cdi/nvidia.yaml";
const IMEX_CONFIG_DIR: &str = "/etc/nvidia-imex";

/// Run nvidia-ctk with given arguments.
fn ctk(args: &[&str]) {
    foreground(NVIDIA_CTK, args);
}

/// Generate CDI spec for GPU device discovery.
/// CDI allows container runtimes (containerd, CRI-O) to inject GPU devices
/// without nvidia-docker. The spec is written to /var/run/cdi/nvidia.yaml
/// where runtimes expect to find it.
pub fn nvidia_ctk_cdi() {
    ctk(&["-d", "cdi", "generate", "--output=/var/run/cdi/nvidia.yaml"]);
    inject_imex_config_mounts();
}

/// nvidia-ctk's nvml discovery does not include /etc/nvidia-imex, but the
/// container's libcuda consults /etc/nvidia-imex/config.cfg during its
/// fabric-readiness check. Without it, cuInit can return
/// CUDA_ERROR_SYSTEM_NOT_READY (802) even when the in-UVM IMEX domain is
/// healthy (nvidia-smi Fabric State=Completed/Success). Add bind mounts for
/// config.cfg + nodes_config.cfg to the generated CDI spec's top-level
/// containerEdits.mounts so they appear inside the pod. Best-effort: warns and
/// leaves the spec unchanged when imex isn't shipped or the spec can't be
/// edited, so it never blocks container start.
fn inject_imex_config_mounts() {
    let present: Vec<String> = ["config.cfg", "nodes_config.cfg"]
        .iter()
        .map(|f| format!("{IMEX_CONFIG_DIR}/{f}"))
        .filter(|p| Path::new(p).exists())
        .collect();
    if present.is_empty() {
        log::info!("nvidia-imex: no {IMEX_CONFIG_DIR} config present; not injecting into CDI");
        return;
    }
    let spec = match fs::read_to_string(CDI_SPEC) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("nvidia-imex: cannot read {CDI_SPEC} to inject config mounts: {e}");
            return;
        }
    };
    match insert_into_toplevel_mounts(&spec, &present) {
        Some(updated) => match fs::write(CDI_SPEC, updated) {
            Ok(()) => log::info!(
                "nvidia-imex: injected {} config mount(s) into {CDI_SPEC}",
                present.len()
            ),
            Err(e) => log::warn!("nvidia-imex: cannot write {CDI_SPEC}: {e}"),
        },
        None => log::warn!(
            "nvidia-imex: top-level containerEdits.mounts not found in {CDI_SPEC}; \
             config not injected into the container"
        ),
    }
}

/// Insert bind-mount entries for `paths` after the top-level
/// `containerEdits: -> mounts:` key of a CDI YAML doc. nvidia-ctk emits
/// 2-space indentation and the top-level containerEdits is the one at column 0
/// (device-level containerEdits blocks are indented). The item indentation is
/// derived from the located `mounts:` line so it survives formatting changes.
/// Returns None if that structure is not present (caller leaves the spec
/// untouched).
fn insert_into_toplevel_mounts(spec: &str, paths: &[String]) -> Option<String> {
    let indent = |l: &str| l.len() - l.trim_start().len();
    let lines: Vec<&str> = spec.lines().collect();

    let ce_idx = lines
        .iter()
        .rposition(|l| l.trim_end() == "containerEdits:" && indent(l) == 0)?;

    let mut mounts_idx = None;
    let mut mounts_indent = 0usize;
    for (i, l) in lines.iter().enumerate().skip(ce_idx + 1) {
        if l.trim().is_empty() {
            continue;
        }
        let ind = indent(l);
        if ind == 0 {
            break; // left the top-level containerEdits block
        }
        if l.trim() == "mounts:" {
            mounts_idx = Some(i);
            mounts_indent = ind;
            break;
        }
    }
    let mounts_idx = mounts_idx?;

    let item = " ".repeat(mounts_indent + 2);
    let field = " ".repeat(mounts_indent + 4);
    let opt = " ".repeat(mounts_indent + 6);
    let mut entries = String::new();
    for p in paths {
        entries.push_str(&format!(
            "{item}- hostPath: {p}\n{field}containerPath: {p}\n{field}options:\n{opt}- ro\n{opt}- nosuid\n{opt}- nodev\n{opt}- bind\n"
        ));
    }

    let mut out = String::new();
    for (i, l) in lines.iter().enumerate() {
        out.push_str(l);
        out.push('\n');
        if i == mounts_idx {
            out.push_str(&entries);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;

    #[test]
    fn test_ctk_fails_without_binary() {
        let result = panic::catch_unwind(|| {
            ctk(&["--version"]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_nvidia_ctk_cdi_fails_without_binary() {
        let result = panic::catch_unwind(|| {
            nvidia_ctk_cdi();
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_into_toplevel_mounts_adds_under_toplevel() {
        let spec = "cdiVersion: \"0.6.0\"
kind: \"nvidia.com/gpu\"
devices:
  - name: \"0\"
    containerEdits:
      deviceNodes:
        - path: /dev/nvidia0
containerEdits:
  deviceNodes:
    - path: /dev/nvidiactl
  mounts:
    - hostPath: /usr/bin/nvidia-smi
      containerPath: /usr/bin/nvidia-smi
      options:
        - ro
        - bind
";
        let paths = vec!["/etc/nvidia-imex/config.cfg".to_string()];
        let out = insert_into_toplevel_mounts(spec, &paths).unwrap();
        assert!(out.contains("- hostPath: /etc/nvidia-imex/config.cfg"));
        assert!(out.contains("containerPath: /etc/nvidia-imex/config.cfg"));
        // original mount is preserved
        assert!(out.contains("/usr/bin/nvidia-smi"));
        // inserted AFTER the top-level mounts: key
        let mounts_pos = out.find("\n  mounts:").unwrap();
        let imex_pos = out.find("/etc/nvidia-imex/config.cfg").unwrap();
        assert!(imex_pos > mounts_pos);
    }

    #[test]
    fn test_insert_into_toplevel_mounts_none_without_mounts() {
        let spec = "kind: \"nvidia.com/gpu\"
containerEdits:
  deviceNodes:
    - path: /dev/nvidiactl
";
        assert!(insert_into_toplevel_mounts(spec, &["/x".to_string()]).is_none());
    }
}
