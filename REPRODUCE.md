# Reproduction guide: PerFace/AVFS-conditioned Arc2Face

Self-contained instructions to reproduce this pipeline from scratch (written
for an agent or human on a fresh GPU server). Goal per `task.md`: an
embedding-to-face pipeline where Arc2Face-style generation is conditioned on a
*perceptual* face embedding (PerFace; optionally AVFS) instead of ArcFace.

**Read the papers first — they are committed in this repo:**
- `2509.20281v1.pdf` — PerFace (Kumagai et al., ICIP 2025): triplet loss
  Eq. (1), training recipe Sec. 4.1, D1/D2 splits, Table 1/3 target numbers.
- `2403.11641.pdf` — Arc2Face (Paraperas Papantoniou et al., ECCV 2024):
  ID-conditioning mechanism Sec. 3.3, original training recipe in the
  supplementary "Implementation Details".

Total compute used originally: ~25 min (Step 1) + ~8 h (Step 2) on one
RTX A6000 48GB. Scaling notes for better hardware at the bottom.

---

## Quickstart (fresh server)

```bash
git clone --recursive https://github.com/weathon/per2face.git && cd per2face
# (if cloned without --recursive: git submodule update --init)

conda create -n per2face python=3.11 -y && conda activate per2face
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

mkdir -p weights checkpoints outputs data

# --- pretrained checkpoints ---
# ArcFace MS1MV3 R50 (mirror of official insightface arcface_torch zoo)
curl -L https://huggingface.co/camenduru/show/resolve/main/models/arcface/ms1mv3_arcface_r50_fp16.pth \
     -o weights/ms1mv3_arcface_r50_fp16.pth         # 174,680,546 bytes
# Arc2Face UNet+encoder and SD1.5 vae/tokenizer/scheduler (into ./weights/hf)
python scripts/download_arc2face.py
# AVFS (optional variant)
curl -L https://zenodo.org/record/7878655/files/avfs_u.pth -o weights/avfs_u.pth

# --- data ---
# 1) simCelebA_triplet.tar.gz + triplet_answers.csv: gdrive folder linked at
#    github.com/kumanotanin/PerFace (triplet_answers.csv is already committed
#    in this repo at data/triplet_answers.csv)
tar -xzf simCelebA_triplet.tar.gz -C data/          # -> data/triplet/*.jpg
# 2) CelebA-HQ 1024x1024 originals (any standard copy), set --images flag

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$(python -c 'import nvidia.cudnn,os;print(os.path.join(os.path.dirname(nvidia.cudnn.__file__),"lib"))')

# --- run order ---
python perface/train_perface.py                                  # Step 1  (~25 min)
python scripts/estimate_template.py   # optional: outputs/triplet_template_224.npy ships with the repo
python scripts/extract_embeddings.py --images <celebahq_dir>     # ~25 min
python scripts/train_arc2face_perface.py --images <celebahq_dir> # ~8 h on A6000
python scripts/generate_samples.py --ckpt checkpoints/arc2face_perface/step016000 \
       --inputs <a few held-out images>
# AVFS variant (optional):
python scripts/extract_avfs_embeddings.py --images <celebahq_dir>
python scripts/train_arc2face_perface.py --images <celebahq_dir> \
       --embs outputs/avfs_embs.npz --out checkpoints/arc2face_avfs
```

## 0. Environment

Python ≥3.10, CUDA GPU. `pip install -r requirements.txt` (see Quickstart for
the torch cu121 index URL). Notes:
- `transformers` must be compatible with Arc2Face's `CLIPTextModelWrapper`
  (uses `transformers.modeling_attn_mask_utils`, i.e. ~4.36).
- insightface's GPU provider needs cuDNN 9 on LD_LIBRARY_PATH; with pip torch
  use the bundled copy (export line in Quickstart). CPU fallback works but is
  ~10x slower for detection.
- Set `HF_HOME` inside the project if you must not write outside it
  (`scripts/download_arc2face.py` does this by default).

Third-party repos are pinned as git submodules at the project root:
- `Arc2Face` (github.com/foivospar/Arc2Face @ 8f3acd7) — we use
  `arc2face/models.py` CLIPTextModelWrapper + `arc2face/utils.py` image_align
- `a_view_from_somewhere` (github.com/SonyResearch/a_view_from_somewhere
  @ 1666c86) — AVFS encoder definition only

## 1. Data and pretrained models

| What | Where | Notes |
|---|---|---|
| SimCelebA triplets (19,200 jpg) | gdrive folder linked in README of github.com/kumanotanin/PerFace → `simCelebA_triplet.tar.gz` | `t####_{a,b,c}.jpg`, 224×224; c = reference |
| `triplet_answers.csv` | same gdrive folder | columns `tripletId,ans1,ans2,ans3`; values a/b |
| ArcFace MS1MV3 R50 | `hf.co/camenduru/show` → `models/arcface/ms1mv3_arcface_r50_fp16.pth` (mirror of official insightface arcface_torch zoo; 174,680,546 bytes) | load into `iresnet50` from insightface `recognition/arcface_torch/backbones/iresnet.py` |
| Arc2Face UNet + encoder | `hf.co/FoivosPar/Arc2Face` subfolders `arc2face/`, `encoder/` | |
| SD1.5 vae/tokenizer/scheduler | `hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5` | |
| AVFS avfs_u (optional) | `zenodo.org/record/7878655/files/avfs_u.pth` | ResNet18, 128-d fc; keep 22 dims per registry in `avfs/build_avfs.py` |
| CelebA-HQ originals 1024² (~28-30k) | any standard CelebA-HQ copy | finetune images |

**Known data defect**: `t4963_c.jpg` is 0 bytes inside the released tarball —
exclude triplet t4963 (handled in `perface/dataset.py:load_annotations`).

## 2. Step 1 — PerFace encoder (face → 512-d perceptual embedding)

Reference: PerFace paper (arXiv 2509.20281), Sec 3.2/4.1.

- Labels: majority of ans1-3 = positive image (x+), other = negative (x-).
  D2 subset = triplets where all three annotators agree (2,128 of 6,400).
- Split: random 80/10/10 at triplet level, seed 42 (matches paper eval
  setting [ii], which is what Table 1's 0.917 corresponds to). Train on
  D2∩train; evaluate only on consistent (D2) val/test triplets.
- Preprocess: resize 224→112 bilinear, (x−127.5)/127.5, random hflip (train).
- Model: iresnet50, init from MS1MV3 ArcFace; finetune ALL params.
- Loss (paper Eq. 1): `max(0, cos(x,x−) − cos(x,x+) + 0.1)` on L2-normalized
  512-d embeddings.
- Optimizer: SGD lr=0.01, momentum=0.9, wd=5e-4, batch 32 triplets, 40 epochs,
  keep best-val checkpoint.

Run: `python perface/train_perface.py` → `checkpoints/perface/best.pth`

**Verification gates** (numbers we got):
- pretrained ArcFace baseline test acc ≈ **0.60** (paper "Original" ≈ 0.60-0.69)
- best val ≈ **0.93** (epoch ~7; converges very fast, loss ~0 by epoch 10)
- test acc ≈ **0.9192** — accept ≥0.85 (paper: 0.917)
- embedding-space health: mean pairwise cos of 500 triplet refs should stay
  ≈0.05±0.13 (same as pretrained ArcFace). If ≈1.0 → space collapsed.

## 3. Step 2 — finetune Arc2Face on PerFace conditioning

### 3.1 How conditioning works (Arc2Face, arXiv 2403.11641 Sec 3.3)

Prompt "photo of a id person" → token embeddings; the `id` token embedding
(position 4) is REPLACED by the face embedding zero-padded 512→768; the
sequence goes through Arc2Face's finetuned CLIP text encoder; the UNet
cross-attends to the (77,768) output. Both text encoder and UNet are
trainable; the face encoder is frozen. CFG uncond branch = empty prompt "".
Implemented in `scripts/a2f_common.py:project_face_embs_train` (differentiable
re-implementation of Arc2Face's `project_face_embs`).

### 3.2 Embedding extraction for the finetune set

`scripts/extract_embeddings.py`. Per CelebA-HQ image:
1. Pad image 300px on all sides (**SCRFD misses tightly-cropped large faces
   without this** — failure rate goes 60% → ~0%), detect with insightface
   buffalo_l, det_size 640, take largest face, subtract pad from kps.
2. Similarity-align 5 kps to the **empirical SimCelebA template**
   (`outputs/triplet_template_224.npy`, created by
   `scripts/estimate_template.py`: mean detected kps over ~800 triplet refs).
   Do NOT use insightface `norm_crop` — the SimSwap-style crop PerFace was
   trained on has visibly different mouth-row placement (~170 vs ~184 px).
3. Warp to 224², resize 112², (x−127.5)/127.5 → PerFace → L2-normalize.
Saves `outputs/perface_embs.npz` (files, embs). ~28k images ≈ 25 min on GPU.

### 3.3 Training

`scripts/train_arc2face_perface.py`:
- Data: CelebA-HQ 1024→512 bilinear, [−1,1]; last 500 sorted files held out.
- Loss: standard eps-prediction MSE on SD1.5 VAE latents (×0.18215),
  uniform t, DDPM schedule from SD1.5 config.
- Conditioning dropout 10% (replace with empty-prompt encoding) for CFG.
- Both UNet and text encoder trainable; bf16 autocast; grad checkpointing
  (non-reentrant for the text encoder!); grad clip 1.0.
- AdamW lr 1e-5 (linear warmup 200), wd 1e-2, batch 16, 16k steps.
  *(Paper stage-2 recipe is lr 1e-6, batch 32, 15 epochs over FFHQ+CelebA-HQ
  ≈47k steps — we used 10× lr and ~1/6 compute because this is re-adaptation
  of a trained model to a nearby embedding space, not fresh conditioning.)*
- Samples every 1k steps on 6 fixed holdout embeddings + reference strip;
  checkpoints (`unet/` + `encoder/` save_pretrained) every 4k.

Inference for checks: DPM-Solver++, 25 steps, CFG 3.0 (paper defaults).

**Verification gates**:
- step 0 grid (pretrained Arc2Face × PerFace embeddings): faces but with
  style drift/artifacts (statues, cartoons, wrong gender) — confirms the
  embedding-space mismatch this finetune fixes.
- loss: ~0.13-0.15 EMA early, very noisy (timestep sampling) — judge by grids.
- by step ~2k: photorealistic portraits, gender consistent, several columns
  clearly inherit reference attributes (skin tone, smile, hair).
- acceptance (task.md): generated faces are plausible portraits visibly
  related to input's perceptual attributes.

### 3.4 Final sampling

`scripts/generate_samples.py --ckpt checkpoints/arc2face_perface/step016000
--inputs <held-out images>` — same detect/align/embed path as 3.2, generates
N samples per input, saves grid with input crop in column 1.

## 4. Optional — AVFS variant

- Encoder: `scripts/avfs_encoder.py` loads avfs_u (ResNet18 → 128 → select 22
  dims). Input: **FFHQ-aligned** 160² (drive Arc2Face's `image_align` with the
  5 SCRFD kps via a synthesized 68-point array — see `ffhq_align_from_kps`),
  then resize 128, center-crop 112, (x−0.5)/0.5.
- **AVFS embeddings are non-negative (SPoSE-style): raw pairwise cos ≈0.78.
  Center on the dataset mean, then L2-normalize** (cos → 0.01±0.45), else the
  conditioning signal is too weak. (`scripts/extract_avfs_embeddings.py`,
  consumed via `a2f_common.load_embs`.)
- Then identical training: same script, `--embs outputs/avfs_embs.npz
  --out checkpoints/arc2face_avfs` (zero-pad 22→768 is handled generically).
- At generation time, subtract the SAME dataset mean stored in the npz.

## 5. Scaling on better hardware

- Match the paper: batch 32+ (grad-accum or more GPUs), lr 1e-6→3e-6,
  40-50k steps, add FFHQ (70k, FFHQ-aligned like CelebA-HQ) to the finetune
  set. Multi-GPU: wrap in accelerate/DDP — the script is single-GPU plain
  PyTorch by design.
- Keep VAE frozen in fp32; bf16 autocast is stable, fp16 needs a scaler.
- Precompute embeddings once (they're frozen-encoder outputs); never put the
  face encoder in the training loop.
- If extending a run: load `unet/` + `encoder/` from the last step dir and
  re-launch with fresh optimizer (no optimizer state is saved).

## 6. Gotchas checklist (things that actually bit us)

1. `t4963_c.jpg` is 0 bytes upstream → skip that triplet.
2. SCRFD + tightly-cropped faces → pad 300px before detection.
3. PerFace crop convention ≠ insightface `norm_crop` → use the empirical
   template from `estimate_template.py`.
4. onnxruntime CUDA needs `libcudnn.so.9` on LD_LIBRARY_PATH (use torch's
   bundled copy).
5. Text-encoder gradient checkpointing: pass `use_reentrant=False` or risk
   silent no-grad.
6. AVFS raw embeddings cluster (cos 0.78) → center before normalizing.
7. transformers >4.36 may break `CLIPTextModelWrapper` imports (attn-mask
   utils moved); pin ~4.36.
8. Train/holdout split is by sorted filename; keep it deterministic so sample
   grids are comparable across runs.
