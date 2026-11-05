#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo 'Usage: bash scripts/ktcal_software_suite.sh MANIFEST NEW_OUTPUT_DIR' >&2
  exit 2
fi
ktcal_manifest=$1
ktcal_output=$2
ktcal_python=${KTCAL_PYTHON:-./.venv/bin/python}
ktcal_device=${KTCAL_DEVICE:-auto}
ktcal_epochs=${KTCAL_EPOCHS:-100}
ktcal_stride=${KTCAL_STRIDE:-1}
if [[ -e "$ktcal_output" ]]; then
  echo 'Output exists; use a new directory. Individual interrupted runs can resume using last.pt.' >&2
  exit 2
fi
mkdir -p "$ktcal_output"

# Freeze all architectures before looking at holdout results.
for ktcal_variant in full no_temporal no_gru full_conv; do
  "$ktcal_python" -B -m ktcal_paper train \
    --manifest "$ktcal_manifest" --output "$ktcal_output/train_$ktcal_variant" \
    --variant "$ktcal_variant" --epochs "$ktcal_epochs" --stride "$ktcal_stride" \
    --val-stride "$ktcal_stride" --device "$ktcal_device" --purpose software-ablation
done

for ktcal_variant in full no_temporal no_gru full_conv; do
  ktcal_mode=ktcal
  if [[ "$ktcal_variant" == full ]]; then ktcal_mode=both; fi
  "$ktcal_python" -B -m ktcal_paper evaluate \
    --manifest "$ktcal_manifest" --checkpoint "$ktcal_output/train_$ktcal_variant/forward_best.pt" \
    --output "$ktcal_output/eval_$ktcal_variant" --mode "$ktcal_mode" --device "$ktcal_device"
done
for ktcal_gate in tighter looser sigma05; do
  case "$ktcal_gate" in
    tighter) ktcal_args=(--tau1 2 --tau2 4 --sigma 1) ;;
    looser) ktcal_args=(--tau1 4 --tau2 6 --sigma 1) ;;
    sigma05) ktcal_args=(--tau1 3 --tau2 5 --sigma .5) ;;
  esac
  "$ktcal_python" -B -m ktcal_paper evaluate \
    --manifest "$ktcal_manifest" --checkpoint "$ktcal_output/train_full/forward_best.pt" \
    --output "$ktcal_output/eval_$ktcal_gate" --mode ktcal --device "$ktcal_device" "${ktcal_args[@]}"
done

"$ktcal_python" -B scripts/collect_ktcal_reports.py "$ktcal_output"
