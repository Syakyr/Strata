#!/bin/sh
# Entry point for the Strata Intel Arc container (Dockerfile.arc, docs/ARC_DOCKER.md).
#
# The SYCL engine is compiled during the image build and lives in the image, so the first start only downloads
# the model. The install config is kept on the /data volume so a recreated container skips the setup pass and
# goes straight to serving - the same shape as docker-entrypoint.sh for the NVIDIA image, with the Intel pieces:
# the setup goes through sycl/setup_intel.py in STRATA_SYCL_NATIVE=1 mode (no Docker inside the container), and
# the start execs sycl/serve/server_intel.py with the config, which is exactly what the generated
# run-<model>.sh would exec.
#
# Setup choices are env vars:
#   STRATA_DATA  where the model files and the config live        (default /data)
#   FAMILY       qwen | swift                                    (default qwen)
#   MODEL        a model setup.py knows                          (default IQ2_XS)
#   CONTEXT      the served context in tokens                     (default 32768)
#   KV           int8 | q4_0 | k8v4; empty keeps setup's default
#   GPUS         "all" or "0,1": one model across several Arc cards (docs/MULTI_GPU.md)
#   LAYER_SPLIT  with GPUS: where each later card's layers start ("24", "24,48"); empty: setup's auto split
#   HOST         0.0.0.0 to answer the network, 127.0.0.1 for this container only (default 0.0.0.0)
#   PORT         the server's port                                (default 8080)
#   API_KEY      required from clients if set - add one before exposing the port beyond localhost
#   REINSTALL    1: run the setup pass again for a model that is already set up (change context, KV, cards)
#
# The Intel port has no vision encoder yet (docs/INTEL_ARC.md), so there is no VISION here: setup_intel.py
# answers "none" for it. --low-ram is forced off there too: the port streams the experts into VRAM, so the
# CUDA engine's low-RAM modes do not apply.
#
# If the container cannot see the cards in /sys/class/drm (some container setups hide it), name them yourself:
#   -e STRATA_INTEL_GPUS="Arc Pro B60:24,Arc Pro B60:24"
set -e
cd /opt/strata || exit 1

# A command passed to `docker run` means the user wants something other than the default setup-and-serve, so
# run exactly that. Before this the entrypoint ignored its arguments entirely: `docker run strata-arc bash`
# fell straight through to the setup pass and started downloading a 29 GB model instead of giving a shell,
# and the CI smoke step did the same thing on its way to a serve attempt. Placed before the GPU guard on
# purpose - a shell inside the container is exactly what you want when the container cannot see a card.
if [ "$#" -gt 0 ]; then
  exec "$@"
fi

STRATA_DATA="${STRATA_DATA:-/data}"
FAMILY="${FAMILY:-qwen}"
MODEL="${MODEL:-IQ2_XS}"
CONTEXT="${CONTEXT:-32768}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
KV="${KV:-}"
GPUS="${GPUS:-}"
LAYER_SPLIT="${LAYER_SPLIT:-}"

# The GPU has to be in the container. Failing here says so in one line instead of failing later inside the
# engine, where the message is a Level Zero error about a missing device.
if [ ! -e /dev/dri ] && [ -z "${STRATA_INTEL_GPUS:-}" ]; then
  echo "No /dev/dri in this container, and STRATA_INTEL_GPUS is not set." >&2
  echo "Start it with the GPU:  docker run --device /dev/dri ..." >&2
  echo "Or name the cards:      -e STRATA_INTEL_GPUS=\"Arc Pro B60:24,Arc Pro B60:24\"" >&2
  exit 1
fi

# setup.py starts the newest strata-*.json it finds, so link in exactly the one this family and model were set
# up with. The config is the recorded output of that setup (the pack, the profile, the quant, the KV decision),
# not settings this script could rebuild from env vars. qwen has an empty family tag.
case "$FAMILY" in qwen) prefix="" ;; *) prefix="${FAMILY}-" ;; esac
tag="${prefix}$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')"
cfg="$STRATA_DATA/config/strata-$tag.json"
mkdir -p "$STRATA_DATA/config"

# The setup pass: downloads the model (~70 GB on first run), packs it, writes the config and the run script.
# Given only when there is no config for this model yet, or REINSTALL=1 - a model already on the volume needs no
# setup pass to start serving.
if [ "${REINSTALL:-0}" = "1" ] || [ ! -f "$cfg" ]; then
  echo "Setting up $tag for an Intel Arc: downloading the model (the engine is already in the image)."
  set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" \
    --data-dir "$STRATA_DATA" --host "$HOST" --api-key "$API_KEY" \
    --port "$PORT" --no-start
  if [ -n "$KV" ]; then set -- "$@" --kv "$KV"; fi
  if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
  if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
  .venv/bin/python sycl/setup_intel.py --setup --yes "$@"
  [ -e "/opt/strata/strata-$tag.json" ] && { cmp -s "/opt/strata/strata-$tag.json" "$cfg" || cp -f "/opt/strata/strata-$tag.json" "$cfg"; }
else
  # The persisted config on the volume is the source of truth, so relink to it on EVERY start. Linking only
  # when /opt/strata/strata-$tag.json is absent is the #1244 bug: a regular file left there by an earlier
  # setup pass survives `docker restart`, the test is then true, and edits to
  # /data/config/strata-$tag.json are silently ignored.
  if [ ! -s "$cfg" ] && [ -f "/opt/strata/strata-$tag.json" ]; then
    cp -f "/opt/strata/strata-$tag.json" "$cfg"      # adopt a config that predates the /data/config layout
  fi
  rm -f "/opt/strata/strata-$tag.json"
  ln -s "$cfg" "/opt/strata/strata-$tag.json"
fi

# Later starts skip straight here. The host and the API key are in the config (setup wrote them), so the server
# needs only the config and the port - the same command the generated run-<model>.sh holds, with the Intel
# server wrapper in place of serve/server.py.
echo "Serving $tag on $HOST:$PORT (engine: $STRATA_SYCL_BIN)."
exec .venv/bin/python sycl/serve/server_intel.py --engine strata --config "$cfg" --port "$PORT"
