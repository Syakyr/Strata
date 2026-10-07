#!/bin/sh
# strata-native.sh: the SYCL-built engine run without Docker, as a drop-in `exe` for
# serve/server.py / sycl/serve/server_intel.py (--engine strata).
#
# sycl/serve/strata-sycl.sh is the same idea but starts the engine inside the
# strata-sycl-dev image with `docker run`. That is right when Strata is installed
# on the host and only the engine needs oneAPI. It is wrong inside a container:
# there is no Docker daemon there, and the oneAPI runtime is already installed.
# This script is the exe for that case - the image built from Dockerfile.arc
# (docs/ARC_DOCKER.md), where the engine, the oneAPI runtime and the Arc compute
# runtime all live in one image and the GPU comes in through --device /dev/dri.
#
# serve/server.py applies the config's "env" block to this process before exec'ing
# it, so what sycl/setup_intel.py wrote into the config (the device selector, the
# mirror size, the verify plan) arrives here as the environment. Everything set
# below is only a default for a key nobody set: `:=` keeps an incoming value.
#
#   STRATA_SYCL_BIN   the engine binary (required; the image sets it)
#   STRATA_HOME       where the install lives, if not /opt/strata
#
# The defaults are the port's runtime notes from docs/INTEL.md:
#   SYCL_CACHE_PERSISTENT=0   the persistent JIT cache crashed on Xe2 during the
#                             first compile. An AOT build needs no JIT at all.
#   ZES_ENABLE_SYSMAN=1       the Level Zero sysfs/telemetry readouts the Monitor
#                             tab and the engine's own VRAM checks use.
#   ONEAPI_DEVICE_SELECTOR=level_zero:*   every Level Zero GPU. `*` rather than a
#                             pinned `:0` so a two-card --layer-split host sees
#                             both cards (the port's own image pins level_zero:0).
set -eu

bin="${STRATA_SYCL_BIN:-}"
if [ -z "$bin" ]; then
    home="${STRATA_HOME:-/opt/strata}"
    for c in "$home/engine-arc/strata" "$home/build-sycl-aot/strata" "$home/build-sycl/strata"; do
        if [ -x "$c" ]; then bin="$c"; break; fi
    done
fi
if [ -z "$bin" ]; then
    echo "strata-native.sh: no engine binary. Set STRATA_SYCL_BIN, or build the" \
         "SYCL engine (sycl/tools/build.sh, docs/INTEL.md)." >&2
    exit 1
fi
if [ ! -x "$bin" ]; then
    echo "strata-native.sh: STRATA_SYCL_BIN=$bin is not an executable file." >&2
    exit 1
fi

: "${SYCL_CACHE_PERSISTENT:=0}"
: "${ZES_ENABLE_SYSMAN:=1}"
: "${ONEAPI_DEVICE_SELECTOR:=level_zero:*}"
export SYCL_CACHE_PERSISTENT ZES_ENABLE_SYSMAN ONEAPI_DEVICE_SELECTOR

exec "$bin" "$@"
