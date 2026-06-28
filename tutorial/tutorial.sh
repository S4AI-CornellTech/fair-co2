#!/usr/bin/env bash
# Fair-CO2 hands-on tutorial runner (see TUTORIAL.md). Self-contained in this repo — wraps tutorial.py.
# Reads only Fair-CO2's committed data (colocation/ref-results, forecast/emb_shapley_lib.py).
#   ./tutorial.sh --workloads exercises/workloads.json
#   ./tutorial.sh --swing llama
#
# Python: set $PYTHON to an interpreter with pandas + numpy (Fair-CO2's env). Defaults to python3.
# From the full-stack-carbon suite, `make tutorial-fairco2` passes the suite's .envs/fair-co2 python.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
exec "$PY" "$HERE/tutorial.py" "$@"
