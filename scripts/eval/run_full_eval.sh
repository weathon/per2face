#!/usr/bin/env bash
# Full eval per eval_guide.md for one model.
# Usage: run_full_eval.sh <encoder perface|avfs> <ckpt dir> <tag> [n_test] [n_attacktrain]
set -euo pipefail
cd "$(dirname "$0")/../.."

ENC="$1"; CKPT="$2"; TAG="$3"
NTEST="${4:-500}"; NTRAIN="${5:-2000}"; BS="${6:-16}"
IMAGES="/home/wg25r/fastdata/marshall/celeba_hq_30k/originals"
PY=/home/wg25r/miniconda/envs/arc2face/bin/python
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/wg25r/miniconda/envs/arc2face/lib/python3.11/site-packages/nvidia/cudnn/lib

# split: test = the 500 held-out (never finetuned); attack-train = disjoint pool
$PY - "$NTRAIN" "$NTEST" <<'EOF'
import json, os, sys
ntrain, ntest = int(sys.argv[1]), int(sys.argv[2])
hold = json.load(open('checkpoints/arc2face_perface/holdout.json'))
allf = sorted(f for f in os.listdir(os.environ.get('IMAGES','/home/wg25r/fastdata/marshall/celeba_hq_30k/originals')) if f.endswith('.jpg'))
hs = set(hold)
pool = [f for f in allf if f not in hs]
open('/tmp/eval_test.txt','w').write("\n".join(hold[:ntest]))
open('/tmp/eval_attacktrain.txt','w').write("\n".join(pool[:ntrain]))
print(f"test={min(ntest,len(hold))} attacktrain={min(ntrain,len(pool))}")
EOF

OUT="outputs/eval/${TAG}"
echo "=== generating test set (held-out) ==="
$PY scripts/eval/build_eval_pairs.py --encoder "$ENC" --ckpt "$CKPT" --images "$IMAGES" \
    --file-list /tmp/eval_test.txt --limit "$NTEST" --out "${OUT}_test" --batch-size "$BS"

echo "=== generating attack-train set ==="
$PY scripts/eval/build_eval_pairs.py --encoder "$ENC" --ckpt "$CKPT" --images "$IMAGES" \
    --file-list /tmp/eval_attacktrain.txt --limit "$NTRAIN" --out "${OUT}_attacktrain" --batch-size "$BS"

echo "=== metrics 1 & 2 (PerFace sim up, ArcFace sim down) ==="
$PY scripts/eval/metrics_sim.py --pairs "${OUT}_test/pairs.npz" | tee "${OUT}_metrics.txt"

echo "=== metric 3 (FaceLinkGen attack) ==="
$PY scripts/eval/attack_facelinkgen.py --train-dir "${OUT}_attacktrain" \
    --test-dir "${OUT}_test" --out "${OUT}_attack.npz" | tee "${OUT}_attack.txt"

echo "=== done: ${OUT}_* ==="
