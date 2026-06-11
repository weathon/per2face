# Eval results (eval_guide.md)

Success criteria: (1) generated face has HIGH PerFace sim to original (look
preserved), (2) LOW ArcFace/insightface sim (identity changed), (3) FaceLinkGen
attack (2602.02914) CANNOT relink the generated face to the original identity.

Held-out test set: 400 celeba_hq identities never seen in finetuning.
FaceLinkGen attack: student trained on 1500 disjoint identities (attacker
replicates the public pipeline). Linkage chance = 1/399 ≈ 0.25% top-1.
ArcFace/PerFace sims are cosine; "match" = ArcFace cos > 0.30.

| Model | PerFace sim (gen,orig) ↑ | vs random | ArcFace sim ↑LOW | % still ID (cos>.3) | Attack top1 (direct / student) ↓ |
|---|---|---|---|---|---|
| **PerFace** (step4000, plain) | 0.55 | +0.51 | **0.385** | **87.5%** | (skipped — known broken) |
| **AVFS** (step16000, plain) | 0.138 | +0.073 | 0.036 | **0.0%** | **1.5% / 0.5%** |
| AVFS + neg-guidance | _pending_ | | | | |
| PerFace + neg-guidance | _pending_ | | | | |

## Read so far

**PerFace conditioning leaks identity.** Because PerFace is finetuned *from*
ArcFace, the embedding still carries identity; Arc2Face faithfully reconstructs
it (gen-vs-orig ArcFace cos 0.385, 87.5% still matching). Great look-preservation
(PerFace sim 0.55), no privacy. Line stopped per task2.md.

**AVFS de-identifies and defeats the attack.** Generated faces are as
identity-dissimilar from the original as two random people (ArcFace cos 0.036 vs
0.012 random; 0% match). The FaceLinkGen student — the attack that broke
frequency-domain PPFR at >98% linkage — recovers only 0.5% top-1 here, at chance.
Cost: perceptual preservation is weak (PerFace sim 0.138, only +0.073 over
random), since the 22-dim AVFS code is coarse. This is the opposite end of the
privacy-utility tradeoff from PerFace.

_Negative-guidance results appended as they complete._
