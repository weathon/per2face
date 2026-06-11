#!/usr/bin/env bash
# Run the full eval matrix sequentially on the free GPU, logging each stage.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=outputs

echo "[1/3] AVFS plain (no neg-guidance)"
bash scripts/eval/run_full_eval.sh avfs checkpoints/arc2face_avfs/step016000 avfs 400 1500 24 \
    > $LOG/eval_avfs_plain.log 2>&1

echo "[2/3] AVFS + negative-guidance CFG"
bash scripts/eval/run_negguide_eval.sh avfs checkpoints/arc2face_avfs/step016000 avfs_ng 400 1500 12 3.0 two_unet \
    > $LOG/eval_avfs_ng.log 2>&1

echo "[3/3] PerFace (step004000) + negative-guidance CFG"
bash scripts/eval/run_negguide_eval.sh perface checkpoints/arc2face_perface/step004000 perface_ng 400 1500 12 3.0 two_unet \
    > $LOG/eval_perface_ng.log 2>&1

echo "ALL EVALS DONE"
