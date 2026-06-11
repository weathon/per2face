# Goal
Build an embedding-to-face generation pipeline where the conditioning embedding is PerFace (soft, perceptual) instead of ArcFace (hard, identity). This is step 1 of a larger privacy project, but for now the only target is: feed in a face, get out a generated face conditioned on its PerFace embedding. No privacy evaluation yet.

# Inputs
- PerFace paper: 2509.20281v1.pdf
- Arc2Face paper: 2403.11641.pdf
- PerFace dataset (triplets): simCelebA_triplet.tar.gz
- Arc2Face repo: ./Arc2Face
- HQ face dataset (for Arc2Face finetuning): ~/fastdata/marshall/celeba_hq/
- Low-res face dataset (for PerFace training, if needed): ~/fastdata/marshall/casia_80k/

# Outputs
1. A trained PerFace encoder: face image -> soft PerFace embedding
2. A finetuned Arc2Face-arch model: PerFace embedding -> face image

# Steps
Change this plan if you think it is needed according to your own judgement and the papers
## Step 1: Reproduce PerFace
1.1. Extract simCelebA_triplet.tar.gz, inspect structure, write a triplet dataloader
1.2. Load pretrained ArcFace (ResNet50, MS1MV3 weights) as the base
1.3. Finetune with triplet loss per Eq. (1) in the PerFace paper (margin=0.1, SGD, lr=0.01, batch=32)
1.4. Train on D2 split (consistent annotations only)
1.5. Evaluate on test triplets, target accuracy ~0.91 per Table 1
Acceptance: test triplet accuracy >= 0.85 (allow some slack from paper's 0.917)

## Step 2: Finetune Arc2Face with PerFace conditioning
2.1. Read Arc2Face repo, identify where ArcFace embedding enters the conditioning path
2.2. Replace the ArcFace encoder with the PerFace encoder from Step 1 (frozen)
2.3. Verify embedding dim compatibility
2.4. Finetune on celeba_hq, keep Arc2Face's original training recipe unless something breaks
2.5. Generate samples: for a held-out face, encode with PerFace, generate, visually inspect
Acceptance: generated faces are plausible portraits and visibly relate to the input face's perceptual attributes (this is qualitative for now, no metrics needed yet)

# Constraints
- Use only GPU 2 (set CUDA_VISIBLE_DEVICES=2) or 1 if 1 has enough free VRAM.
- Do not modify files outside the current directory
- No destructive actions (no rm -rf on data dirs, no force-push, no overwriting pretrained weights without backup)
- Use git for version control, commit after each sub-step completes
- If a base model needs downloading (ArcFace MS1MV3, Arc2Face checkpoint, SD base), do it, but log the source

# Notes
- User will be away from laptop. For blocking questions or when a step completes, send push notification to phone
- Save intermediate checkpoints; don't only keep the final one
- Be mindful about disk space