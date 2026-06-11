"""Extract AVFS (avfs_u) embeddings for celeba_hq images.

Per image: SCRFD detect (padded) -> FFHQ-align from 5 kps to 160x160 ->
resize 128 / center-crop 112 / (x-0.5)/0.5 -> avfs_u -> 22-dim embedding.

Saves raw embeddings plus the dataset mean; the training representation is
centered + L2-normalized (AVFS lives in the positive orthant, raw cosines
are ~0.8-1.0, so centering is needed for a usable conditioning signal).
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image

from avfs_encoder import ffhq_align_from_kps, load_avfs_u, preprocess

from insightface.app import FaceAnalysis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="/home/wg25r/fastdata/marshall/celeba_hq_30k/originals")
    p.add_argument("--out", default="outputs/avfs_embs.npz")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    app = FaceAnalysis(name="buffalo_l", root="weights/insightface",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    net = load_avfs_u().cuda()

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                   glob.glob(os.path.join(args.images, "*.png")))
    print(f"{len(paths)} images")

    PAD = 300
    files, embs, skipped = [], [], []
    batch, batch_files = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).cuda()
        with torch.no_grad():
            e = net(x).float()
        embs.append(e.cpu().numpy())
        files.extend(batch_files)
        batch.clear()
        batch_files.clear()

    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            skipped.append(path)
            continue
        padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
        faces = app.get(padded)
        if not faces:
            skipped.append(path)
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        kps = face.kps.astype(np.float32) - PAD
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        aligned = ffhq_align_from_kps(pil, kps)
        batch.append(preprocess(aligned))
        batch_files.append(os.path.basename(path))
        if len(batch) >= args.batch_size:
            flush()
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(paths)} done, {len(skipped)} skipped", flush=True)

    flush()
    embs = np.concatenate(embs).astype(np.float32)
    mean = embs.mean(0)
    np.savez(args.out, files=np.array(files), embs_raw=embs, mean=mean)

    en = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    c = embs - mean
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    idx = np.random.default_rng(0).choice(len(embs), 1000, replace=False)
    for name, e in [("raw", en[idx]), ("centered", cn[idx])]:
        S = e @ e.T
        off = S[~np.eye(len(e), dtype=bool)]
        print(f"{name}: pairwise cos mean={off.mean():.3f} std={off.std():.3f}")
    print(f"saved {args.out}: {embs.shape}, skipped {len(skipped)}")


if __name__ == "__main__":
    main()
