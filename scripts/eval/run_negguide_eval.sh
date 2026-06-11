#!/usr/bin/env bash
# Negative-guidance eval (task2.md): generate de-id faces with CFG negative =
# the person's true identity via original Arc2Face, then run the same metrics +
# FaceLinkGen attack as the plain eval. Attacker replicates the full public
# pipeline (incl. the CFG trick), so attack-train is also neg-guidance-generated.
#
# Usage: run_negguide_eval.sh <perface|avfs> <ckpt dir> <tag> [n_test] [n_train] [bs] [guidance] [neg_mode]
set -euo pipefail
cd "$(dirname "$0")/../.."

ENC="$1"; CKPT="$2"; TAG="$3"
NTEST="${4:-400}"; NTRAIN="${5:-1500}"; BS="${6:-12}"; G="${7:-3.0}"; NEG="${8:-two_unet}"
IMAGES="/home/wg25r/fastdata/marshall/celeba_hq_30k/originals"
PY=/home/wg25r/miniconda/envs/arc2face/bin/python
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/wg25r/miniconda/envs/arc2face/lib/python3.11/site-packages/nvidia/cudnn/lib

$PY - "$NTRAIN" "$NTEST" <<'EOF'
import json, os, sys
ntrain, ntest = int(sys.argv[1]), int(sys.argv[2])
hold = json.load(open('checkpoints/arc2face_perface/holdout.json'))
allf = sorted(f for f in os.listdir('/home/wg25r/fastdata/marshall/celeba_hq_30k/originals') if f.endswith('.jpg'))
hs = set(hold); pool = [f for f in allf if f not in hs]
open('/tmp/ng_test.txt','w').write("\n".join(hold[:ntest]))
open('/tmp/ng_attacktrain.txt','w').write("\n".join(pool[:ntrain]))
print(f"test={min(ntest,len(hold))} attacktrain={min(ntrain,len(pool))}")
EOF

OUT="outputs/eval/${TAG}"
echo "=== neg-guidance generate: test (held-out) [neg=$NEG g=$G] ==="
$PY scripts/eval/generate_negguide.py --encoder "$ENC" --ckpt "$CKPT" --neg-mode "$NEG" \
    --images "$IMAGES" --file-list /tmp/ng_test.txt --limit "$NTEST" \
    --out "${OUT}_test" --batch-size "$BS" --guidance "$G"

echo "=== neg-guidance generate: attack-train ==="
$PY scripts/eval/generate_negguide.py --encoder "$ENC" --ckpt "$CKPT" --neg-mode "$NEG" \
    --images "$IMAGES" --file-list /tmp/ng_attacktrain.txt --limit "$NTRAIN" \
    --out "${OUT}_attacktrain" --batch-size "$BS" --guidance "$G"

echo "=== metrics 1 & 2 ==="
$PY scripts/eval/metrics_sim.py --pairs "${OUT}_test/pairs.npz" | tee "${OUT}_metrics.txt"

echo "=== metric 3 (FaceLinkGen attack) ==="
$PY scripts/eval/attack_facelinkgen.py --train-dir "${OUT}_attacktrain" \
    --test-dir "${OUT}_test" --out "${OUT}_attack.npz" | tee "${OUT}_attack.txt"

echo "=== done: ${OUT}_* ==="
