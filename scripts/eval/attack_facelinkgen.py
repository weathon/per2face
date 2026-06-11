"""Metric 3 from eval_guide.md: reproduce the FaceLinkGen attack (2602.02914)
against our de-identification, to measure how much identity leaks.

Threat model adapted: the released "protected template" is our GENERATED
(de-identified) face. The attacker has oracle access to the public pipeline,
so they build (original, generated) pairs on attacker-controlled identities
and train a student f_s to recover the ArcFace identity of the ORIGINAL from
the generated face (cosine distillation, paper Eq. 2). They then:

  Linkage  (Sec 4.1): for each test generated face, nearest-neighbour of
    f_s(generated) against the database of original ArcFace embeddings.
    Reported as top-1 recall + Success@5; chance = 1/N.
  Direct baseline: same, but query = ArcFace(generated) with no student
    (what you get by just running a face recognizer on the released face).

Privacy is strong if linkage recall stays near chance / the direct baseline
stays low. A big student-over-direct gain means the de-id is invertible.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from insightface.utils.face_align import norm_crop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import PROJ, detect, get_app
sys.path.insert(0, os.path.join(PROJ, "perface"))
from iresnet import iresnet50


def load_crops(gen_dir, files, arc_orig, app):
    """ArcFace-aligned 112 crops of generated faces, paired with original
    ArcFace embeddings. Drops faces that fail detection."""
    crops, targets = [], []
    for f, t in zip(files, arc_orig):
        bgr = cv2.imread(os.path.join(gen_dir, f))
        if bgr is None:
            continue
        face = detect(app, bgr)
        if face is None:
            continue
        # norm_crop wants kps in the original (unpadded) frame, which detect() returns
        crop = norm_crop(bgr, face.kps, image_size=112)
        crops.append(crop[:, :, ::-1].copy())
        targets.append(t)
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float()
    x = (x - 127.5) / 127.5
    return x, torch.from_numpy(np.stack(targets)).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True, help="attack-train eval dir (gen/ + pairs.npz)")
    p.add_argument("--test-dir", required=True, help="held-out test eval dir")
    p.add_argument("--arcface-w", default="weights/ms1mv3_arcface_r50_fp16.pth")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = "cuda"
    app = get_app()

    def load(d):
        z = np.load(os.path.join(d, "pairs.npz"), allow_pickle=True)
        ok = z["gen_ok"].astype(bool)
        return z["files"][ok], z["arc_orig"][ok], z["arc_gen"][ok]

    tr_files, tr_arc_o, _ = load(args.train_dir)
    te_files, te_arc_o, te_arc_g = load(args.test_dir)
    print(f"attack-train pairs: {len(tr_files)}, test pairs: {len(te_files)}", flush=True)

    Xtr, Ytr = load_crops(os.path.join(args.train_dir, "gen"), tr_files, tr_arc_o, app)
    Xte, Yte = load_crops(os.path.join(args.test_dir, "gen"), te_files, te_arc_o, app)
    Ytr = F.normalize(Ytr, dim=1)
    print(f"student train crops: {len(Xtr)}, test crops: {len(Xte)}", flush=True)

    # student f_s: ArcFace-arch, initialised from ArcFace, distilled to map
    # generated face -> original identity (paper Eq. 2)
    student = iresnet50(fp16=False)
    student.load_state_dict(torch.load(args.arcface_w, map_location="cpu", weights_only=True))
    student.to(device).train()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    n = len(Xtr)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x, y = Xtr[idx].to(device), Ytr[idx].to(device)
            e = F.normalize(student(x).float(), dim=1)
            loss = (1 - (e * y).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  cos-dist loss={tot / n:.4f}", flush=True)

    student.eval()
    with torch.no_grad():
        q_student = []
        for i in range(0, len(Xte), 128):
            q_student.append(F.normalize(student(Xte[i:i + 128].to(device)).float(), dim=1).cpu())
        q_student = torch.cat(q_student).numpy()

    db = te_arc_o / (np.linalg.norm(te_arc_o, axis=1, keepdims=True) + 1e-9)  # original identities
    N = len(db)
    gt = np.arange(N)

    def linkage(query, tag):
        q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
        S = q @ db.T
        order = np.argsort(-S, axis=1)
        top1 = (order[:, 0] == gt).mean()
        top5 = np.array([gt[i] in order[i, :5] for i in range(N)]).mean()
        same_sim = S[gt, gt].mean()
        print(f"  {tag:32s} top1={top1*100:5.1f}%  success@5={top5*100:5.1f}%  "
              f"mean cos to true={same_sim:.3f}")
        return top1, top5

    print(f"\nLinkage attack ({N} test identities, chance top1={100/N:.2f}%):")
    d1, d5 = linkage(te_arc_g, "direct: ArcFace(generated)")
    s1, s5 = linkage(q_student, "FaceLinkGen student f_s(generated)")
    print(f"\n  privacy read: student top1={s1*100:.1f}% (direct {d1*100:.1f}%). "
          f"{'STRONG protection' if s1 < 0.05 else 'PARTIAL' if s1 < 0.3 else 'WEAK - identity recoverable'}")

    if args.out:
        np.savez(args.out, q_student=q_student, db=db,
                 direct_top1=d1, direct_top5=d5, student_top1=s1, student_top5=s5)


if __name__ == "__main__":
    main()
