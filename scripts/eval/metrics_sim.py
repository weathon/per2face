"""Metrics 1 & 2 from eval_guide.md, on a pairs.npz built by build_eval_pairs.

1. PerFace sim(generated, original)  -> want HIGH (perceptual look preserved)
2. ArcFace sim(generated, original)  -> want LOW  (identity changed)

Reports the paired diagonal vs a random-different-pair baseline, so we can
say both "more perceptually similar than two random faces" and "as
identity-dissimilar as two random faces".
"""
import argparse

import numpy as np


def paired_and_random(a, b, rng):
    """a,b: (N,D) L2-normalized. Returns (paired diag, random off-diag)."""
    paired = np.sum(a * b, axis=1)
    perm = rng.permutation(len(b))
    perm[perm == np.arange(len(b))] = (perm[perm == np.arange(len(b))] + 1) % len(b)
    random = np.sum(a * b[perm], axis=1)
    return paired, random


def summ(name, x):
    print(f"  {name:38s} mean={x.mean():.3f}  std={x.std():.3f}  "
          f"med={np.median(x):.3f}  [{np.percentile(x,5):.3f}, {np.percentile(x,95):.3f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--arcface-thresh", type=float, default=0.30,
                   help="cos above which two faces count as the same identity")
    args = p.parse_args()

    d = np.load(args.pairs, allow_pickle=True)
    ok = d["gen_ok"].astype(bool)
    n_total = len(ok)
    arc_o, arc_g = d["arc_orig"][ok], d["arc_gen"][ok]
    per_o, per_g = d["per_orig"][ok], d["per_gen"][ok]
    rng = np.random.default_rng(0)

    print(f"pairs: {int(ok.sum())}/{n_total} (generated face detectable)\n")

    print("Metric 1 - PerFace sim (generated vs original)  [want HIGH]")
    per_pair, per_rand = paired_and_random(per_g, per_o, rng)
    summ("generated-vs-its-original", per_pair)
    summ("generated-vs-random-original", per_rand)
    print(f"  -> paired is {per_pair.mean() - per_rand.mean():+.3f} above random "
          f"({'GOOD: look preserved' if per_pair.mean() > per_rand.mean() + 0.05 else 'WEAK'})\n")

    print("Metric 2 - ArcFace/insightface sim (generated vs original)  [want LOW]")
    arc_pair, arc_rand = paired_and_random(arc_g, arc_o, rng)
    summ("generated-vs-its-original", arc_pair)
    summ("random-different-identities", arc_rand)
    match_rate = float((arc_pair > args.arcface_thresh).mean())
    print(f"  -> {match_rate*100:.1f}% of generated faces still match the original "
          f"identity at cos>{args.arcface_thresh}")
    print(f"  -> paired is {arc_pair.mean() - arc_rand.mean():+.3f} vs random-different "
          f"({'GOOD: identity changed' if arc_pair.mean() < args.arcface_thresh else 'LEAKS identity'})\n")

    # combined: privacy-utility per face
    good = (per_pair > per_rand.mean()) & (arc_pair < args.arcface_thresh)
    print(f"Both (high PerFace AND low ArcFace): {good.mean()*100:.1f}% of faces")


if __name__ == "__main__":
    main()
