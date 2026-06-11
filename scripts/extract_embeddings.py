"""Extract PerFace embeddings for celeba_hq images.

Pipeline per image: SCRFD detection (buffalo_l) -> similarity-align the
5-point landmarks to the empirical SimCelebA triplet template at 224
(outputs/triplet_template_224.npy) -> resize 112 -> PerFace iresnet50 ->
L2-normalized 512-dim embedding.

Saves outputs/perface_embs.npz: files (N,), embs (N,512) float32.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perface"))
from iresnet import iresnet50

from insightface.app import FaceAnalysis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="/home/wg25r/fastdata/marshall/celeba_hq_30k/originals")
    p.add_argument("--weights", default="checkpoints/perface/best.pth")
    p.add_argument("--template", default="outputs/triplet_template_224.npy")
    p.add_argument("--out", default="outputs/perface_embs.npz")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    template = np.load(args.template).astype(np.float32)

    app = FaceAnalysis(name="buffalo_l", root="weights/insightface",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    net = iresnet50(fp16=False)
    net.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    net.eval().cuda()

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                   glob.glob(os.path.join(args.images, "*.png")))
    print(f"{len(paths)} images")

    files, embs, skipped = [], [], []
    batch_crops, batch_files = [], []

    def flush():
        if not batch_crops:
            return
        x = torch.from_numpy(np.stack(batch_crops)).permute(0, 3, 1, 2).float().cuda()
        x = (x - 127.5) / 127.5
        with torch.no_grad():
            e = torch.nn.functional.normalize(net(x).float(), dim=1)
        embs.append(e.cpu().numpy())
        files.extend(batch_files)
        batch_crops.clear()
        batch_files.clear()

    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            skipped.append(path)
            continue
        faces = app.get(img)
        if not faces:
            skipped.append(path)
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        M, _ = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), template,
                                           method=cv2.LMEDS)
        crop = cv2.warpAffine(img, M, (224, 224), borderValue=0)
        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        batch_crops.append(crop[:, :, ::-1].copy())  # BGR -> RGB
        batch_files.append(os.path.basename(path))
        if len(batch_crops) >= args.batch_size:
            flush()
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(paths)} done, {len(skipped)} skipped", flush=True)

    flush()
    embs = np.concatenate(embs).astype(np.float32)
    np.savez(args.out, files=np.array(files), embs=embs)
    print(f"saved {args.out}: {embs.shape}, skipped {len(skipped)}")
    if skipped:
        with open(args.out + ".skipped.txt", "w") as f:
            f.write("\n".join(skipped))


if __name__ == "__main__":
    main()
