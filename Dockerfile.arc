# syntax=docker/dockerfile:1
#
# Strata on an Intel Arc (the SYCL port in sycl/, docs/INTEL_ARC.md) as one self-contained image.
#
# This is the deployable Intel image. The other Intel image in this repo, sycl/tools/Dockerfile, is a
# *development* image: it has the compiler and the migration tools, and the engine it builds is started by
# sycl/serve/strata-sycl.sh, which needs a Docker daemon on the host (docker-in-docker). Here the engine, the
# oneAPI runtime and the Arc compute runtime are all in one image, the GPU comes in through --device /dev/dri,
# and nothing shells out to Docker: sycl/setup_intel.py runs in STRATA_SYCL_NATIVE=1 mode and the engine's exe
# is sycl/serve/strata-native.sh.
#
# The engine is compiled during the build, so the first container start only downloads the model and serves.
# The build needs no GPU: the AOT path (ocloc) cross-compiles for the named device on the CPU.
#
# Build:
#   docker build -f Dockerfile.arc -t strata-arc .                    # AOT for the Arc Pro B60 (bmg-g21)
#   docker build -f Dockerfile.arc --build-arg SYCL_AOT=bmg-g31 -t strata-arc .    # Arc Pro B70
#   docker build -f Dockerfile.arc --build-arg SYCL_AOT= -t strata-arc .           # SPIR-V, JIT at run time
#
# Run (host: an Intel GPU the kernel's xe driver drives, and /dev/dri/renderD*):
#   docker run --rm --device /dev/dri -p 8080:8080 -v strata-arc-data:/data \
#     -e MODEL=IQ2_XS -e CONTEXT=32768 strata-arc
#
# Two cards, one model (docs/MULTI_GPU.md; the 2x B60 numbers are in
# bench/results/2026-10-04-community-2x-arc-pro-b60):
#   docker run --rm --device /dev/dri -p 8080:8080 -v strata-arc-data:/data \
#     -e MODEL=IQ1_M -e GPUS=all -e LAYER_SPLIT=24 strata-arc
#
# Setup choices are env vars read by docker-entrypoint-arc.sh: FAMILY, MODEL, CONTEXT, KV, GPUS, LAYER_SPLIT,
# HOST, PORT, API_KEY. Only the model files, the pack, the MTP layer and the install config live in /data; the
# engine is part of the image. See docs/ARC_DOCKER.md for the whole flow, the card table and the troubleshooting
# notes.
#
# Pinned versions (verified against the two repositories on 2026-10-07; bump them together - the SYCL runtime,
# MKL and the GPU compute runtime are tested as a set, and docs/INTEL.md's numbers were made with oneAPI
# 2026.1.1 + NEO 26.09):
#   oneAPI compiler/runtime  https://apt.repos.intel.com/oneapi  (Packages index: apt-cache policy after add)
#   Arc compute runtime      https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics
ARG BASE_IMAGE=ubuntu:24.04

# --------------------------------------------------------------------------------------------- builder
FROM ${BASE_IMAGE} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG ONEAPI_COMPILER=2026.1.1-325
ARG ONEAPI_MKL=2026.1.0-236
ARG OCLOC=26.31.39395.14-1~24.04~ppa1
# The AOT device. bmg-g21 = Arc Pro B60 / B580 / B570 (docs/INTEL_ARC.md: the B60's PCI ids e211/e221 are
# filed with the BMG-G21 cards and `ocloc ids bmg-g21` prints 20.1.0, which is what the card reports).
# bmg-g31 = Arc Pro B70. Empty means SPIR-V only: the runtime JITs every kernel on first use (~47 s per the
# port's notes, again on every start, because SYCL_CACHE_PERSISTENT=0 is required on Xe2).
ARG SYCL_AOT=bmg-g21
ARG BUILD_PARITY=OFF
ARG JOBS=4

# The Intel GPU compute runtime is not in the oneAPI repository: ocloc (the ahead-of-time GPU compiler) and the
# Level Zero / OpenCL userspace driver come from the kobuk-team PPA Intel points at for client GPUs.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
         -o /tmp/oneapi.key \
    && gpg --batch --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg /tmp/oneapi.key \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
         > /etc/apt/sources.list.d/oneAPI.list \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x0c0e6af955ce463c03fc51574d098d70afbe5e1f" \
         -o /tmp/intel-gpu.key \
    && gpg --batch --dearmor -o /usr/share/keyrings/intel-gpu-keyring.gpg /tmp/intel-gpu.key \
    && echo "deb [signed-by=/usr/share/keyrings/intel-gpu-keyring.gpg arch=amd64] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" \
         > /etc/apt/sources.list.d/intel-graphics.list \
    && rm -f /tmp/oneapi.key /tmp/intel-gpu.key

RUN apt-get update && apt-get install -y --no-install-recommends \
        intel-oneapi-compiler-dpcpp-cpp=${ONEAPI_COMPILER} \
        intel-oneapi-mkl-devel=${ONEAPI_MKL} \
        intel-ocloc=${OCLOC} \
        cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && icpx --version | head -2 \
    && command -v ocloc cmake ninja git \
    && ls -l /usr/bin/ocloc        # the PPA ships ocloc-<ver> and links it with update-alternatives

WORKDIR /opt/strata
COPY . .

# The engine only, configured the way sycl/tools/build.sh does it. STRATA_SYCL_PARITY=OFF skips the 150-odd
# kernel test and bench targets: they run on the card, and the card is not here. FetchContent pulls llama.cpp at
# the pinned commit for ggml, so the build needs the network.
RUN set -ex \
    && . /opt/intel/oneapi/setvars.sh --force > /dev/null \
    && cmake -S sycl -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx \
        -DSTRATA_SYCL_AOT=${SYCL_AOT} \
        -DSTRATA_SYCL_PARITY=${BUILD_PARITY} \
    && cmake --build build --target strata -j ${JOBS} \
    && test -x build/strata \
    && ls -l build/strata

# --------------------------------------------------------------------------------------------- runtime
FROM ${BASE_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG ONEAPI_RUNTIME=2026.1.1-325
ARG ONEAPI_MKL_RUNTIME=2026.1.0-236
ARG COMPUTE_RUNTIME=26.31.39395.14-1~24.04~ppa1
ARG LIBZE=1.32.0-1~24.04~ppa1

ENV PYTHONUNBUFFERED=1 LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
         -o /tmp/oneapi.key \
    && gpg --batch --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg /tmp/oneapi.key \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
         > /etc/apt/sources.list.d/oneAPI.list \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x0c0e6af955ce463c03fc51574d098d70afbe5e1f" \
         -o /tmp/intel-gpu.key \
    && gpg --batch --dearmor -o /usr/share/keyrings/intel-gpu-keyring.gpg /tmp/intel-gpu.key \
    && echo "deb [signed-by=/usr/share/keyrings/intel-gpu-keyring.gpg arch=amd64] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" \
         > /etc/apt/sources.list.d/intel-graphics.list \
    && rm -f /tmp/oneapi.key /tmp/intel-gpu.key

# intel-opencl-icd is not needed by a Level Zero run, but it is the same NEO build and it is what makes `clinfo`
# and any OpenCL-flavoured tool work inside the container; it costs one layer of the same driver already there.
RUN apt-get update && apt-get install -y --no-install-recommends \
        intel-oneapi-runtime-dpcpp-cpp=${ONEAPI_RUNTIME} \
        intel-oneapi-runtime-mkl=${ONEAPI_MKL_RUNTIME} \
        libze1=${LIBZE} \
        libze-intel-gpu1=${COMPUTE_RUNTIME} \
        intel-opencl-icd=${COMPUTE_RUNTIME} \
        python3 python3-pip python3-venv libgomp1 libatomic1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Neither oneAPI runtime package drops an ld.so.conf.d entry, so add the trees ourselves. Discovered from what is
# actually installed rather than hardcoded: the runtime package layout is not the compiler package layout, and the
# first version of this step assumed /opt/intel/oneapi/compiler/*/lib, which the runtime image does not have.
#
# The `if` form is load-bearing, not style. The previous line was
#     [ -d "$d" ] && echo "$d" >> conf
# and when the last glob in the list missed, that `&&` list returned 1, the `for` loop returned 1, and the outer
# `&& ldconfig` never ran - the RUN died with exit 1 before ldconfig, which reads like a missing library and is
# really a shell exit-status bug.
RUN set -eu \
    && conf=/etc/ld.so.conf.d/strata-oneapi.conf \
    && : > "$conf" \
    && find /opt/intel -maxdepth 5 -type d -name lib 2>/dev/null \
         | while IFS= read -r d; do \
              if ls "$d"/*.so* >/dev/null 2>&1; then printf '%s\n' "$d"; fi; \
            done \
         | sort -u >> "$conf" \
    && echo "--- $conf ---" && cat "$conf" \
    && ldconfig \
    && missing=0 \
    && for want in libsycl.so libmkl_sycl_blas libze_loader; do \
         if ldconfig -p | grep -q "$want"; then echo "ok    $want"; \
         else echo "MISS  $want"; missing=1; fi; \
       done \
    && if [ "$missing" -ne 0 ]; then \
         echo "FATAL: a library the engine links against is not resolvable in the runtime image" >&2; \
         echo "where things actually are:" >&2; \
         find /opt/intel -maxdepth 4 -type d 2>/dev/null | head -60 >&2; \
         exit 1; \
       fi

WORKDIR /opt/strata
COPY . .

# The port's runtime environment (docs/INTEL.md), declared before the checks below so they see what the
# container will actually run with. ONEAPI_DEVICE_SELECTOR=level_zero:* so a two-card host sees both cards;
# SYCL_CACHE_PERSISTENT=0 because the persistent JIT cache crashed on Xe2; ZES_ENABLE_SYSMAN=1 for the VRAM
# readouts; STRATA_VERIFY_DEVICE_PLAN=1 for the device-built verify plan.
# STRATA_VERIFY_NO_HOST=1 is deliberately NOT set here: it is only valid when every expert is resident in VRAM,
# and on a card that mirrors part of them in RAM that path is the one that has hung. Add it with -e when the
# engine's own startup line says all the experts are resident.
ENV STRATA_SYCL_NATIVE=1 \
    STRATA_SYCL_BIN=/opt/strata/engine-arc/strata \
    ONEAPI_DEVICE_SELECTOR=level_zero:* \
    SYCL_CACHE_PERSISTENT=0 \
    ZES_ENABLE_SYSMAN=1 \
    STRATA_VERIFY_DEVICE_PLAN=1

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && chmod +x docker-entrypoint-arc.sh sycl/serve/strata-native.sh

COPY --from=builder /opt/strata/build/strata /opt/strata/engine-arc/strata

# The build-time proof that the runtime package set is complete: every shared object the engine needs resolves,
# and the binary runs far enough to print its usage (which happens before any device is opened). The full check
# is sycl/tools/test_arc_image.py, which runs in CI and can be run on the Arc box before trusting the image.
RUN set -ex \
    && if ldd /opt/strata/engine-arc/strata | grep -q 'not found'; then \
         ldd /opt/strata/engine-arc/strata | grep 'not found'; exit 1; \
       fi \
    && /opt/strata/engine-arc/strata --help > /dev/null \
    && .venv/bin/python sycl/tools/test_arc_image.py

VOLUME ["/data"]
EXPOSE 8080

# /health is answered before the API key gate, so it works with or without one. The port only opens after the
# model loads (1-3 minutes, longer on a first run), so a too short start period would mark a still-loading
# container unhealthy and a restart policy would kill it mid-download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint-arc.sh"]
