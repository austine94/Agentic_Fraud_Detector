#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec python -u -m experiments.benchmark_latency_quantiles
