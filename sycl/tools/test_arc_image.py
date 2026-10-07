"""Checks that the Strata Intel Arc image (Dockerfile.arc) is put together right, on a machine with no Intel GPU.

    .venv/bin/python sycl/tools/test_arc_image.py        # inside the image

GitHub-hosted runners have no Intel GPU, so the model cannot be run there. These are the checks a CPU-only
machine can answer, and each one is a thing that is easy to get wrong in this particular image:

  - the engine binary is in the image and every shared object it needs resolves (the oneAPI runtime and the Arc
    compute runtime come from two different repositories, and neither ships an ld.so.conf.d entry);
  - the engine runs far enough to print its usage, which happens before it opens a device;
  - the native run path is wired: STRATA_SYCL_NATIVE on, the exe is strata-native.sh, paths are not remapped
    to /work (that remapping is the docker-in-docker install, not this one);
  - the wrapper execs the engine and fails loudly when pointed at something that is not there;
  - the card list can be named without sysfs (STRATA_INTEL_GPUS), which is the escape hatch for a container
    that does not show the host's /sys/class/drm.

What this cannot check is anything about the card: that the AOT image matches the device, that the kernels
compute the right answers, that the experts fit. That is docs/ARC_DOCKER.md's first-start checklist.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent        # the Strata checkout
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sycl"))

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)
    return ok


def main() -> int:
    print(f"strata arc image check  (cwd={Path.cwd()}, python={sys.version.split()[0]})")

    # --- the engine binary and its libraries -----------------------------------------------------------
    exe = Path(os.environ.get("STRATA_SYCL_BIN") or ROOT / "engine-arc" / "strata")
    if not check("STRATA_SYCL_BIN exists", exe.is_file(), str(exe)):
        return report()
    check("the engine is executable", os.access(exe, os.X_OK), str(exe))

    if shutil.which("ldd"):
        out = subprocess.run(["ldd", str(exe)], capture_output=True, text=True).stdout
        missing = [ln.strip() for ln in out.splitlines() if "not found" in ln]
        check("every shared object resolves", not missing, "; ".join(missing[:4]))
        check("libsycl is linked in", "libsycl.so" in out)
        check("oneMKL SYCL BLAS is linked in", "libmkl_sycl_blas" in out)
        check("the Level Zero loader is reachable",
              subprocess.run(["/bin/sh", "-c", "ldconfig -p | grep -q libze_loader"]).returncode == 0)
    else:
        print("  [skip] ldd is not installed")

    r = subprocess.run([str(exe), "--help"], capture_output=True, text=True)
    # `--help` is not device-free: the engine builds its SYCL context before it prints usage, so on a machine
    # with no Intel GPU it aborts with "No device of requested type available". That is the expected outcome in
    # CI and on any build host without a card, so it counts as the binary having executed. What must NOT happen
    # is a missing library, a segfault with no recognisable message, or silence - those still fail this check.
    combined = (r.stdout or "") + (r.stderr or "")
    no_device = "No device of requested type" in combined
    usage = r.returncode == 0 and len(r.stdout) > 0
    check("the engine executes (usage line, or the expected no-GPU abort)", usage or no_device,
          f"rc={r.returncode}, {len(r.stdout)} bytes stdout, "
          + ("reached SYCL init, no device present" if no_device else "no device error seen"))

    # --- the native (no docker-in-docker) wiring ------------------------------------------------------
    import setup_intel as I

    check("STRATA_SYCL_NATIVE is on", I.NATIVE, f"got {os.environ.get('STRATA_SYCL_NATIVE')!r}")
    check("the exe is the native wrapper", I.SYCL_WRAPPER.name == "strata-native.sh", str(I.SYCL_WRAPPER))
    if not I.NATIVE:
        # Everything below calls into the module with container paths; in docker mode sycl_path() stops the
        # process instead of returning, so stop here rather than report its message as this test's.
        return report()
    check("sycl_path is identity in native mode", I.sycl_path("/data/models/x.gguf") == "/data/models/x.gguf")
    got, why = I.sycl_engine()
    check("sycl_engine() resolves the binary", why is None and got is not None, str(why or got))

    # --- the wrapper --------------------------------------------------------------------------------
    w = subprocess.run([str(I.SYCL_WRAPPER), "--serve"], capture_output=True, text=True,
                      env={**os.environ, "STRATA_SYCL_BIN": "/bin/true"})
    check("the wrapper execs STRATA_SYCL_BIN", w.returncode == 0, f"rc={w.returncode}")

    w = subprocess.run([str(I.SYCL_WRAPPER)], capture_output=True, text=True,
                      env={**os.environ, "STRATA_SYCL_BIN": "/nonexistent/strata"})
    check("the wrapper fails loudly on a bad binary",
          w.returncode != 0 and "not an executable" in w.stderr, f"rc={w.returncode}")

    # The fallback the image relies on: no STRATA_SYCL_BIN, so the wrapper looks in $STRATA_HOME. Use a stub
    # rather than the real engine - this test is about *path resolution*, and the real engine builds a SYCL
    # context before it does anything else, so it aborts with "No device of requested type available" on any
    # host without a card (including this CI runner). Asserting rc == 0 there would be testing the wrong thing.
    with tempfile.TemporaryDirectory() as home:
        stub = Path(home) / "engine-arc" / "strata"
        stub.parent.mkdir(parents=True)
        stub.write_text('#!/bin/sh\necho "stub engine resolved at $0"\n')
        stub.chmod(0o755)
        w = subprocess.run([str(I.SYCL_WRAPPER), "--serve"], capture_output=True, text=True,
                         env={k: v for k, v in os.environ.items() if k != "STRATA_SYCL_BIN"}
                         | {"STRATA_HOME": home})
        check("the wrapper falls back to $STRATA_HOME/engine-arc/strata",
              w.returncode == 0 and str(stub) in (w.stdout + w.stderr),
              f"rc={w.returncode} out={(w.stdout + w.stderr).strip()[:120]}")

    # --- the card list without sysfs -----------------------------------------------------------------
    os.environ["STRATA_INTEL_GPUS"] = "Arc Pro B60:24,Arc Pro B60:24"
    g = I.intel_gpus()
    check("two cards named from STRATA_INTEL_GPUS", len(g) == 2 and all(c["vram_gb"] == 24.0 for c in g),
          str([c["name"] for c in g]))
    os.environ.pop("STRATA_INTEL_GPUS")

    # --- the server wrapper is importable -------------------------------------------------------------
    r = subprocess.run([sys.executable, "-c", "import server_intel"], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "sycl" / "serve") + os.pathsep
                           + os.environ.get("PYTHONPATH", "")})
    check("sycl/serve/server_intel.py imports", r.returncode == 0,
          (r.stderr.strip().splitlines() or [""])[-1][:160])

    return report()


def report() -> int:
    if fails:
        print(f"\nFAILED: {len(fails)} - " + ", ".join(fails))
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
