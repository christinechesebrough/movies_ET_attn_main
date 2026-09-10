#!/usr/bin/env bash
# Fan extract_fooof_light.py across recordings, one pinned worker per slice.
#
# BLAS threads MUST stay at 1: FOOOF is thousands of tiny curve_fit calls and
# an unpinned numpy spawns a thread pool per call. Measured 8.6 fits/sec
# unpinned vs ~82 fits/sec pinned on real spectra.
#
# Usage:  bash analysis_scripts/launch_fooof_light.sh [n_workers]
# Logs:   $LOGDIR/fooof_light_w<N>.log
# Stop:   pkill -f extract_fooof_light.py

set -u
N=${1:-16}
PY=/home/christine/anaconda3/envs/mne310/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=${LOGDIR:-"${CLAUDE_JOB_DIR:-/tmp}/tmp"}
mkdir -p "$LOGDIR"

echo "launching $N workers -> $LOGDIR"
for ((w=0; w<N; w++)); do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  FOOOF_WORKER=$w FOOOF_N_WORKERS=$N \
    nohup "$PY" -u "$HERE/extract_fooof_light.py" \
    > "$LOGDIR/fooof_light_w${w}.log" 2>&1 &
  echo "  worker $w  pid $!"
done
wait
