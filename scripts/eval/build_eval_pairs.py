"""Generate de-identified faces for held-out originals and collect embeddings.

For each original image: condition the finetuned Arc2Face model on its
PerFace/AVFS embedding, generate a face, then embed BOTH original and
generated with PerFace and ArcFace(insightface). Saves generated images and
an npz of L2-normalized embeddings for the similarity + attack evals.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (PROJ, AVFSEmb, PerFaceEmb, detect, get_app,
                         load_pipeline)

sys.path.insert(0, os.path.join(PROJ, "scripts"))
from a2f_common import build_prompt_ids, project_face_embs_train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["perface", "avfs"], required=True)
    p.add_argument("--ckpt", required=True, help="stepXXXXXX dir or 'pretrained'")
    p.add_argument("--images", required=True, help="dir of original face images")
    p.add_argument("--file-list", help="optional txt of filenames to restrict to")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--perface-w", default="checkpoints/perface/best.pth")
    p.add_argument("--template", default="outputs/triplet_template_224.npy")
    p.add_argument("--avfs-w", default="weights/avfs_u.pth")
    p.add_argument("--avfs-mean", default="outputs/avfs_embs.npz")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    device = "cuda"
    os.makedirs(os.path.join(args.out, "gen"), exist_ok=True)
    app = get_app()
    perface = PerFaceEmb(args.perface_w, args.template, device)
    cond = perface if args.encoder == "perface" else AVFSEmb(args.avfs_w, args.avfs_mean, device)
    pipe = load_pipeline(args.ckpt, device)
    cond_ids, uncond_ids, id_pos = build_prompt_ids(pipe.tokenizer, device)

    if args.file_list:
        names = [l.strip() for l in open(args.file_list) if l.strip()]
    else:
        names = sorted(os.listdir(args.images))
    names = [n for n in names if n.lower().endswith((".jpg", ".png"))]
    names = names[args.offset:args.offset + args.limit]

    recs = []  # (file, cond_emb, arc_orig, per_orig)
    for n in names:
        bgr = cv2.imread(os.path.join(args.images, n))
        if bgr is None:
            continue
        f = detect(app, bgr)
        if f is None:
            continue
        arc_orig = f.embedding / (np.linalg.norm(f.embedding) + 1e-9)
        c = cond.crop(bgr, f.kps)
        per_c = perface.crop(bgr, f.kps)
        recs.append({"file": n, "cond_crop": c, "per_crop": per_c, "arc_orig": arc_orig})

    print(f"{len(recs)} originals detected", flush=True)

    # conditioning + per_orig embeddings
    cond_embs = cond.embed_crops([r["cond_crop"] for r in recs]).cpu().numpy()
    per_orig = perface.embed_crops([r["per_crop"] for r in recs]).cpu().numpy()

    # generation (batched)
    gen_paths = []
    bs = args.batch_size
    for i in range(0, len(recs), bs):
        emb = torch.from_numpy(cond_embs[i:i + bs]).to(device)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            pe = project_face_embs_train(pipe.text_encoder, emb, cond_ids, uncond_ids, id_pos)
            neg = project_face_embs_train(
                pipe.text_encoder, emb, cond_ids, uncond_ids, id_pos,
                drop_mask=torch.ones(len(emb), dtype=torch.bool, device=device))
            imgs = pipe(prompt_embeds=pe, negative_prompt_embeds=neg,
                        num_inference_steps=args.steps, guidance_scale=args.guidance,
                        generator=torch.Generator(device).manual_seed(args.seed)).images
        for j, im in enumerate(imgs):
            path = os.path.join(args.out, "gen", recs[i + j]["file"])
            im.save(path)
            gen_paths.append(path)
        print(f"generated {min(i + bs, len(recs))}/{len(recs)}", flush=True)

    # embed generated faces (detect first)
    arc_gen = np.zeros_like(per_orig[:, :512]) if False else None
    arc_gen_list, per_gen_crops, gen_ok = [], [], []
    for r, gp in zip(recs, gen_paths):
        bgr = cv2.imread(gp)
        f = detect(app, bgr)
        if f is None:
            gen_ok.append(False)
            arc_gen_list.append(np.zeros(512, np.float32))
            per_gen_crops.append(np.zeros((112, 112, 3), np.uint8))
            continue
        gen_ok.append(True)
        arc_gen_list.append(f.embedding / (np.linalg.norm(f.embedding) + 1e-9))
        per_gen_crops.append(perface.crop(bgr, f.kps))
    arc_gen = np.stack(arc_gen_list).astype(np.float32)
    per_gen = perface.embed_crops(per_gen_crops).cpu().numpy()

    files = np.array([r["file"] for r in recs])
    np.savez(os.path.join(args.out, "pairs.npz"),
             files=files, arc_orig=np.stack([r["arc_orig"] for r in recs]).astype(np.float32),
             per_orig=per_orig.astype(np.float32), arc_gen=arc_gen,
             per_gen=per_gen.astype(np.float32), gen_ok=np.array(gen_ok),
             cond_embs=cond_embs.astype(np.float32))
    print(f"saved {args.out}/pairs.npz  ({len(files)} pairs, {int(np.sum(gen_ok))} gen detected)")


if __name__ == "__main__":
    main()
