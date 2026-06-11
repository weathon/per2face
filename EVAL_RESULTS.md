# Eval results (eval_guide.md)

Success criteria: (1) generated face has HIGH PerFace sim to original (look
preserved), (2) LOW ArcFace/insightface sim (identity changed), (3) FaceLinkGen
attack (2602.02914) CANNOT relink the generated face to the original identity.

Held-out test set: 400 celeba_hq identities never seen in finetuning.
FaceLinkGen attack: student trained on 1500 disjoint identities (attacker
replicates the public pipeline, incl. the CFG trick). Linkage chance ≈ 0.25%
top-1. Sims are cosine; "match" = ArcFace cos > 0.30. "Neg-guidance" = CFG with
the negative condition set to the person's true identity via original Arc2Face
(task2.md), to push generation away from the real identity.

| Model | PerFace sim ↑ (look) | ArcFace sim ↓ (id) | % still ID | Attack top1 ↓ direct / student | verdict |
|---|---|---|---|---|---|
| PerFace plain (step4000)      | **0.55**  (+0.51) | 0.385 | 87.5% | (skipped; ≥ ng's 63%) | leaks |
| PerFace + neg-guidance        | 0.405 (+0.355)| 0.264 | 36.6% | **63.4%** / 7.4%       | still leaks |
| **AVFS plain (step16000)**    | 0.138 (+0.073)| 0.036 | **0.0%** | **1.5%** / 0.5%      | **protects** |
| AVFS + neg-guidance           | 0.105 (+0.045)| 0.026 | 0.0%  | 0.0% / 1.3%            | protects |

(PerFace sim and ArcFace sim are mean cosine of generated-vs-its-original;
"+x" is the margin over a random different original. Attack = linkage top-1.)

Visual comparison (`outputs/eval/comparison_grid.jpg`, columns: original,
PerFace+ng, AVFS, AVFS+ng): PerFace+ng clearly reproduces the *same* person as
the original (look preserved, identity leaks); AVFS columns are *different*
people who share only coarse attributes (de-identified).

## Conclusion

**AVFS is the privacy winner; PerFace cannot be made private here.**

- **AVFS de-identifies and defeats the attack.** Generated faces are as
  identity-dissimilar from the original as two random people (ArcFace cos 0.036
  ≈ 0.012 random; 0% match). The FaceLinkGen attack — which broke frequency-
  domain PPFR at >98% linkage — gets 1.5% direct / 0.5% student, at chance. The
  cost is weak look-preservation (PerFace sim 0.138, +0.073 over random): the
  22-dim AVFS code keeps only coarse perceptual attributes.

- **PerFace conditioning leaks identity, and neg-guidance can't fix it.**
  Because PerFace is finetuned *from* ArcFace, its embedding still encodes
  identity and Arc2Face reconstructs it (plain: ArcFace cos 0.385, 87.5% match).
  Negative-guidance (pushing away from the true identity) roughly halves the
  leak (→ 0.264, 36.6% match) while keeping good look (0.405), but a plain
  face recognizer still relinks **63.4%** of identities. Partial mitigation, not
  protection.

- **Negative-guidance is only useful where there is leakage to cancel.** It
  helps PerFace (87.5%→36.6% match) but does nothing for AVFS (already at
  random) except slightly reduce resemblance.

- **Methodology note.** When the released artifact is itself a face image, the
  *direct* attack (run ArcFace on the generated face) is the binding privacy
  metric — it is already near-optimal. The FaceLinkGen distillation student is
  the stronger attack only when the protected template is NOT directly
  embeddable (the original paper's frequency-domain setting); here the student
  does not beat direct (PerFace+ng: 7.4% vs 63.4%).

## Privacy–utility tradeoff (the core finding)

The two embeddings sit at opposite ends. PerFace maximizes resemblance but
preserves identity (no privacy); AVFS maximizes privacy but preserves little
resemblance. Neither yields "looks similar AND different identity AND
attack-proof" at high quality on both axes simultaneously — AVFS is the only one
that satisfies the privacy criteria (2 & 3), at the cost of criterion 1's
strength. Next directions to close the gap: a perceptual embedding with more
capacity than AVFS-22 but explicitly identity-stripped (unlike PerFace), or
training the generator with an identity-dissimilarity loss rather than relying
on the embedding alone.
