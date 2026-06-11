# PerFace → Arc2Face: perceptual-embedding-conditioned face generation

Step 1 of a privacy project: generate faces conditioned on a **PerFace**
(soft, perceptual) embedding instead of **ArcFace** (hard, identity).
See `task.md` for the full task description.

## Pipeline

1. **PerFace encoder** (`perface/`): ArcFace iresnet50 (MS1MV3) finetuned with
   the PerFace triplet loss (Eq. 1, margin 0.1) on the SimCelebA triplet
   dataset (D2 = consistent annotations).
   `face image (aligned 112x112) -> 512-dim L2-normalized embedding`
2. **Arc2Face finetune** (`scripts/`): the Arc2Face UNet + CLIP-encoder
   (SD1.5-based) finetuned on CelebA-HQ-30k, conditioned on frozen PerFace
   embeddings via the `<id>` pseudo-prompt token (zero-padded 512→768).
   `PerFace embedding -> 512x512 face image`

## Results

| Step | Metric | Value |
|---|---|---|
| PerFace (D2) | test triplet accuracy | **0.9192** (paper: 0.917, accept ≥0.85) |
| | pretrained-ArcFace baseline | 0.60 |

**Privacy eval (eval_guide.md): see [EVAL_RESULTS.md](EVAL_RESULTS.md).** Headline:
AVFS conditioning de-identifies and defeats the FaceLinkGen attack (1.5% top-1
linkage, at chance) but preserves only coarse resemblance; PerFace conditioning
preserves resemblance but leaks identity (relinkable) even with negative-guidance
CFG. Privacy–utility tradeoff, AVFS on the privacy end.

## Key files

- `perface/train_perface.py` — Step 1 training (SGD lr 0.01, m 0.9, wd 5e-4, bs 32)
- `perface/dataset.py` — SimCelebA triplet loader (majority vote = positive)
- `scripts/estimate_template.py` — empirical 5-pt landmark template of the triplet crops
- `scripts/extract_embeddings.py` — celeba_hq -> aligned crop -> PerFace embedding
- `scripts/a2f_common.py` — differentiable Arc2Face conditioning projection
- `scripts/train_arc2face_perface.py` — Step 2 finetune (UNet + text encoder, bf16,
  AdamW 1e-5, CFG dropout 0.1, noise-pred MSE)
- `scripts/generate_samples.py` — input face -> PerFace -> generated grid

## Data / model sources

- SimCelebA triplets + `triplet_answers.csv`: gdrive linked from
  github.com/kumanotanin/PerFace (t4963 has a 0-byte image; excluded)
- ArcFace MS1MV3 R50: `hf.co/camenduru/show` `models/arcface/ms1mv3_arcface_r50_fp16.pth`
  (mirror of official insightface arcface_torch model zoo)
- `perface/iresnet.py`: github.com/deepinsight/insightface (MIT)
- Arc2Face UNet + encoder: `hf.co/FoivosPar/Arc2Face`
- SD1.5 VAE/tokenizer/scheduler: `hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5`
- Face detection: insightface buffalo_l (SCRFD); images padded 300px before
  detection (SCRFD misses tightly-cropped large faces)

## Environment

conda env `arc2face` (torch 2.4.1+cu121, diffusers 0.29.2, transformers 4.36.0,
insightface 0.7.3). GPU 2 only (`CUDA_VISIBLE_DEVICES=2`). For GPU onnxruntime:
`LD_LIBRARY_PATH=.../site-packages/nvidia/cudnn/lib`.

## Alignment convention

PerFace inputs must be aligned like the SimCelebA crops: similarity-transform
the 5 SCRFD landmarks to `outputs/triplet_template_224.npy` (224x224), then
resize to 112x112. This template differs from insightface's standard arcface
template (shorter chin margin) — do not use `norm_crop`.
