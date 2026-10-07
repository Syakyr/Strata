# Strata in a container, on an Intel Arc

One image with the SYCL engine, the oneAPI runtime and the Arc compute runtime in it. The GPU comes in through
`--device /dev/dri`; nothing inside the container needs a Docker daemon.

This is a different thing from `sycl/tools/Dockerfile`, which is the port's *development* image: it carries the
compiler and the migration tools, and the engine it builds is started by `sycl/serve/strata-sycl.sh`, which runs
it with `docker run` and mounts the folder above the checkout at `/work`. That is the right shape when Strata is
installed on the host and only the engine needs oneAPI. It is the wrong shape for a machine that just wants to run
the model: you cannot nest that container inside itself. Here the engine runs in the container's own environment
(`STRATA_SYCL_NATIVE=1`), and its `exe` is `sycl/serve/strata-native.sh` instead of `strata-sycl.sh`.

Everything about the port itself — what is fast, what is not, what has been measured on which card — is in
[INTEL_ARC.md](INTEL_ARC.md) and [INTEL.md](INTEL.md). This page is only about the image.

## The host needs

- **Docker** (or Podman) with `--device /dev/dri` support. The image ships the *userspace* compute runtime; the
  **kernel driver is the host's** and cannot be shipped in an image.
- **A kernel whose `xe` driver drives your card.** For Battlemage (Arc B-series, Pro B50/B60/B70) on Ubuntu 24.04
  that means the hardware-enablement (HWE) kernel — Ubuntu Desktop 24.04 tracks HWE by default, a GA-kernel install
  does not. Check with `lspci -nn -d 8086:` (you should see the card) and `ls /sys/bus/pci/drivers/xe` (the driver
  bound to it). If the kernel does not drive the card, no container can.
- **VRAM for the model.** A 24 GB card holds the Coder IQ1_M or the Flash-Next IQ2_XS with every expert resident;
  see the model table in [INTEL_ARC.md](INTEL_ARC.md).
- **Disk.** The model is ~70 GB and the pack is written next to it. Give the `/data` volume room.

The image does not need oneAPI, the PPA or the driver installed on the host. It needs the *kernel* driver, and
nothing else.

## Run it

```sh
docker run --rm --device /dev/dri -p 8080:8080 \
  -v strata-arc-data:/data \
  -e MODEL=IQ2_XS -e CONTEXT=32768 \
  ghcr.io/<owner>/strata-arc:0.1.40-bmg-g21
```

The first start downloads the model (70 GB, minutes to an hour depending on the link) and then serves. Every start
after that goes straight to serving: the config is on the volume and the engine is in the image.

Watch it come up:

```sh
docker logs -f <container>          # the engine's own lines: what it put where, and what fits
curl -s localhost:8080/health       # answered before the API key gate, so it works with or without a key
```

Then point anything OpenAI-compatible at `http://localhost:8080/v1` — the same API, streaming, tool calls and web
app as the NVIDIA and AMD engines.

### Two cards, one model

```sh
docker run --rm --device /dev/dri -p 8080:8080 \
  -v strata-arc-data:/data \
  -e MODEL=IQ1_M -e GPUS=all -e LAYER_SPLIT=24 \
  ghcr.io/<owner>/strata-arc:0.1.40-bmg-g21
```

`GPUS=all` takes every card the setup found; `LAYER_SPLIT=24` says the second card's layers start at layer 24 —
it is a layer number, not layers per card. The two-B60 measurements in
[`bench/results/2026-10-04-community-2x-arc-pro-b60`](../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md)
used 24 for Coder IQ1_M and 25 for Flash-Next IQ2_XS, with every expert resident on both cards. Leaving
`LAYER_SPLIT` empty lets the setup place the split itself from each card's free VRAM; pin it when you know better.

On a split, the startup line that says "N experts are neither in VRAM nor mirrored" counts the *other* card's
layers. The `100% of the experts resident` lines are the ones to read.

### The environment

| Variable | Default | What it does |
|---|---|---|
| `MODEL` | `IQ2_XS` | the model to serve (any model `setup.py` knows) |
| `FAMILY` | `qwen` | `qwen` or `swift` |
| `CONTEXT` | `32768` | served context in tokens |
| `KV` | setup's default | `int8`, `q4_0` or `k8v4` |
| `GPUS` | all found | `all`, or `0,1` |
| `LAYER_SPLIT` | auto | where each later card's layers start |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | where the server listens |
| `API_KEY` | none | required from clients if set — **add one before the port is reachable from anywhere you do not trust** |
| `REINSTALL` | off | `1` re-runs the setup pass for a model that is already set up (change the context, the KV, the cards) |
| `STRATA_DATA` | `/data` | where the model files, the pack and the config live |
| `STRATA_INTEL_GPUS` | probe sysfs | name the cards yourself: `"Arc Pro B60:24,Arc Pro B60:24"` |

The engine's own switches are ordinary environment variables and pass straight through the entry point and the
wrapper to the engine, so `-e STRATA_MIRROR_MIB=6144` or `-e STRATA_STAGE_TRIM=1` work as they do on a bare
install. `STRATA_VERIFY_NO_HOST=1` is deliberately **not** set in the image: it is only valid when every expert
is resident in VRAM, and on a card that mirrors part of them in RAM that is the path that has hung. Set it once
the startup lines tell you the experts are all resident.

## Build it

```sh
docker build -f Dockerfile.arc -t strata-arc .                     # AOT for the Arc Pro B60 (bmg-g21)
docker build -f Dockerfile.arc --build-arg SYCL_AOT=bmg-g31 .     # Arc Pro B70
docker build -f Dockerfile.arc --build-arg SYCL_AOT= .             # SPIR-V, JIT at run time
```

The build needs no GPU: the ahead-of-time path cross-compiles with `ocloc` on the CPU. It does need the network —
CMake's `FetchContent` pulls llama.cpp at the pinned commit for ggml.

**Build the AOT one.** Without it the runtime JIT-compiles every kernel on first use, about 47 s per the port's
notes, and again on every start, because `SYCL_CACHE_PERSISTENT=0` is required on Xe2 (the persistent JIT cache
crashed during the first compile). AOT means the container starts in the time the model takes to load and nothing
compiles.

`bmg-g21` is the right target for the Arc Pro B60: its PCI ids `e211`/`e221` are filed with the BMG-G21 cards
and `ocloc ids bmg-g21` prints 20.1.0, which is what the card reports.

The image is two stages. The builder installs `intel-oneapi-compiler-dpcpp-cpp`, `intel-oneapi-mkl-devel` and
`intel-ocloc` and builds only the `strata` target (`STRATA_SYCL_PARITY=OFF` — the kernel tests run on a card,
and the card is not there). The runtime installs `intel-oneapi-runtime-dpcpp-cpp`, `intel-oneapi-runtime-mkl`
and the Arc compute runtime (`libze1`, `libze-intel-gpu1`, `intel-opencl-icd`) and copies the engine in. The
compiler packages come from Intel's oneAPI apt repository; `ocloc` and the compute runtime are **not** in it and
come from the [`kobuk-team/intel-graphics`](https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics)
PPA that Intel's client-GPU guide points at. The versions are pinned in the Dockerfile as a set, because the SYCL
runtime, MKL and the compute runtime are tested as a set — the numbers in [INTEL.md](INTEL.md) were made with
oneAPI 2026.1.1 and NEO 26.09.

Neither oneAPI runtime package writes an `ld.so.conf.d` entry, so the image writes one itself from the trees that
actually got installed, and then fails the build if any shared object the engine needs does not resolve.

### In CI

`.github/workflows/docker-arc.yml` builds the image on a CPU-only GitHub-hosted runner and pushes it to GHCR as
`ghcr.io/<owner>/strata-arc:<version>-<aot>`. Run it from the **Run workflow** button: pick the AOT device,
whether to push, whether to also tag `:latest`.

A runner has no Intel GPU, so CI cannot run the model. What it does check is `sycl/tools/test_arc_image.py`,
which you can also run inside the image on your own machine before trusting it:

```sh
docker run --rm ghcr.io/<owner>/strata-arc:0.1.40-bmg-g21 \
  .venv/bin/python sycl/tools/test_arc_image.py
```

It checks that every shared object the engine needs resolves, that the engine runs as far as its usage line
(which happens before it opens a device), that the native run path is wired (the exe is the native wrapper,
paths are not remapped to `/work`), that the wrapper execs the engine and fails loudly when pointed at something
that is not there, and that the card list can be named without sysfs.

## When it does not work

**"No /dev/dri in this container."** You did not pass `--device /dev/dri`. Add it.

**"no Intel Arc found (an xe or i915 card in /sys/class/drm)".** The container cannot see the card in sysfs.
Some container setups hide it. Name the cards yourself: `-e STRATA_INTEL_GPUS="Arc Pro B60:24,Arc Pro B60:24"`.
If the *host* also does not see the card under the `xe` driver, the host kernel is the problem, not the image.

**The engine starts and dies with a Level Zero error.** The userspace driver in the image and the kernel driver on
the host disagree. Check `dmesg | grep -i xe` on the host for a driver that refused the card, and try a newer HWE
kernel. The compute runtime in the image can be bumped without touching anything else in the engine.

**It starts slowly and the log shows compiling.** The image was built without AOT (`SYCL_AOT=`). Rebuild with the
device that matches your card.

**It hangs on a card that mirrors experts in RAM.** If you set `STRATA_VERIFY_NO_HOST=1`, unset it — that switch
is only valid when every expert is resident.

**The container is killed while loading.** The setup sizes the KV cache from the RAM it measures, and inside a
container `/proc/meminfo` is the *host's* total, not the container's limit. If you cap the container's memory,
lower `CONTEXT` until the KV fits, or give the container the memory the context needs.

**The port is open and there is no key.** `HOST` defaults to `0.0.0.0` so the container is reachable from the
network. Set `API_KEY` before you publish the port anywhere you do not control.

## What is verified and what is not

The image was written against the port's own documented build and run procedure, and against the two package
repositories, checked on 2026-10-07: the oneAPI compiler/runtime/MKL versions, that `ocloc` and the compute
runtime are not in the oneAPI repository and are in the PPA, that the PPA's `intel-ocloc` links
`/usr/bin/ocloc` through `update-alternatives`, that the SYCL runtime package lands `libsycl.so.9` and the
Level Zero adapters in `/opt/intel/oneapi/compiler/2026.1/lib`, that the MKL runtime lands
`libmkl_sycl_blas.so.6` in `/opt/intel/oneapi/redist/lib`, and that neither writes an `ld.so.conf.d` entry.
The native-mode config transform and the wrapper are covered by checks that run locally and in CI.

**Not verified: that this image has ever run on an Arc.** There is no Intel GPU in the build environment, and the
image has not been built here either. The first build is the CI run; the first run on a card is yours. The
performance numbers quoted above are the community's, measured without Docker
([`bench/results/2026-10-04-community-2x-arc-pro-b60`](../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md));
a container adds a `--device` passthrough and nothing else on the GPU path, but nobody has put a stopwatch on
that yet. If you run it, the report format in `docs/INTEL_ARC.md`'s "Reporting a problem" section is what moves
this from experimental to supported.
