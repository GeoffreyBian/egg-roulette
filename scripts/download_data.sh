#!/usr/bin/env bash
# Human glove data: OpenGraph Labs, Ego-Tactile Manipulation (CC-BY-4.0), ~18 MB.
set -euo pipefail
cd "$(dirname "$0")/.."
base=https://huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation/resolve/main
mkdir -p data/ogl
curl -fL "$base/data/chunk-000/file-000.parquet" -o data/ogl/data.parquet
