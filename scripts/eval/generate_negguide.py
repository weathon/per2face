"""Negative-guidance generation (task2.md idea).

Classifier-free guidance where the NEGATIVE condition is the person's true
identity (original face -> ArcFace -> original Arc2Face), so generation is
actively pushed away from the real identity while being pulled toward our
perceptual (PerFace/AVFS) target:

    eps = eps_neg + s * (eps_pos - eps_neg)

  eps_pos: our de-id model (UNet + encoder) conditioned on the perceptual emb
  eps_neg:
    --neg-mode two_unet  -> original Arc2Face UNet+encoder on the real ArcFace
                            identity (most faithful to "original face on Arc2Face")
    --neg-mode our_unet  -> our UNet, but conditioned on the original ArcFace
                            identity projected through the original encoder

Writes gen/ images + pairs.npz in the same schema as build_eval_pairs, so the
existing metrics_sim / attack_facelinkgen run on it unchanged.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (PROJ, AVFSEmb, PerFaceEmb, detect, get_app,
                         load_pipeline)
sys.path.insert(0, os.path.join(PROJ, "scripts"))
from a2f_common import build_prompt_ids, project_face_embs_train


@torch.no_grad()
def sample(scheduler, latents, steps, scale, unet_pos, prompt_pos,
           unet_neg, prompt_neg, device):
    scheduler.set_timesteps(steps, device=device)
    latents = latents * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        li = scheduler.scale_model_input(latents, t)
        with torch.autocast("cuda", torch.bfloat16):
            eps_pos = unet_pos(li, t, encoder_hidden_states=prompt_pos).sample
            eps_neg = unet_neg(li, t, encoder_hidden_states=prompt_neg).sample
        eps = eps_neg + scale * (eps_pos - eps_neg)
        latents = scheduler.step(eps, t, latents).prev_sample
    return latents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["perface", "avfs"], required=True)
    p.add_argument("--ckpt", required=True, help="our de-id model stepXXXXXX dir")
    p.add_argument("--neg-mode", choices=["two_unet", "our_unet"], default="two_unet")
    p.add_argument("--images", required=True)
    p.add_argument("--file-list", required=True)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--out", required=True)
    p.add_argument("--perface-w", default="checkpoints/perface/best.pth")
    p.add_argument("--template", default="outputs/triplet_template_224.npy")
    p.add_argument("--avfs-w", default="weights/avfs_u.pth")
    p.add_argument("--avfs-mean", default="outputs/avfs_embs.npz")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    device = "cuda"
    os.makedirs(os.path.join(args.out, "gen"), exist_ok=True)

    app = get_app()
    perface = PerFaceEmb(args.perface_w, args.template, device)
    cond_enc = perface if args.encoder == "perface" else AVFSEmb(args.avfs_w, args.avfs_mean, device)

    pos_pipe = load_pipeline(args.ckpt, device)            # our de-id model
    neg_pipe = load_pipeline("pretrained", device)         # original Arc2Face
    vae = pos_pipe.vae
    scheduler = pos_pipe.scheduler
    cond_ids, uncond_ids, id_pos = build_prompt_ids(pos_pipe.tokenizer, device)

    unet_pos = pos_pipe.unet
    unet_neg = neg_pipe.unet if args.neg_mode == "two_unet" else pos_pipe.unet
    enc_pos = pos_pipe.text_encoder       # our (perceptual) encoder
    enc_neg = neg_pipe.text_encoder       # original Arc2Face (identity) encoder

    names = [l.strip() for l in open(args.file_list) if l.strip()][:args.limit]

    recs = []
    for n in names:
        bgr = cv2.imread(os.path.join(args.images, n))
        if bgr is None:
            continue
        f = detect(app, bgr)
        if f is None:
            continue
        recs.append({
            "file": n,
            "arc_orig": f.embedding / (np.linalg.norm(f.embedding) + 1e-9),
            "cond_crop": cond_enc.crop(bgr, f.kps),
            "per_crop": perface.crop(bgr, f.kps),
        })
    print(f"{len(recs)} originals detected", flush=True)

    cond_embs = cond_enc.embed_crops([r["cond_crop"] for r in recs]).cpu().numpy()
    per_orig = perface.embed_crops([r["per_crop"] for r in recs]).cpu().numpy()
    arc_orig = np.stack([r["arc_orig"] for r in recs]).astype(np.float32)

    gen_paths = []
    bs = args.batch_size
    for i in range(0, len(recs), bs):
        emb_pos = torch.from_numpy(cond_embs[i:i + bs]).to(device)
        emb_neg = torch.from_numpy(arc_orig[i:i + bs]).to(device)  # true ArcFace identity
        b = len(emb_pos)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            prompt_pos = project_face_embs_train(enc_pos, emb_pos, cond_ids, uncond_ids, id_pos)
            prompt_neg = project_face_embs_train(enc_neg, emb_neg, cond_ids, uncond_ids, id_pos)
        g = torch.Generator(device).manual_seed(args.seed + i)
        latents = torch.randn((b, 4, 64, 64), generator=g, device=device, dtype=torch.float32)
        latents = sample(scheduler, latents, args.steps, args.guidance,
                         unet_pos, prompt_pos, unet_neg, prompt_neg, device)
        with torch.no_grad():
            imgs = vae.decode(latents / vae.config.scaling_factor).sample
        imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        for j in range(b):
            from PIL import Image
            path = os.path.join(args.out, "gen", recs[i + j]["file"])
            Image.fromarray(imgs[j]).save(path)
            gen_paths.append(path)
        print(f"generated {min(i + bs, len(recs))}/{len(recs)}", flush=True)

    # embed generated faces
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

    np.savez(os.path.join(args.out, "pairs.npz"),
             files=np.array([r["file"] for r in recs]), arc_orig=arc_orig,
             per_orig=per_orig.astype(np.float32), arc_gen=arc_gen,
             per_gen=per_gen.astype(np.float32), gen_ok=np.array(gen_ok),
             cond_embs=cond_embs.astype(np.float32))
    print(f"saved {args.out}/pairs.npz ({len(recs)} pairs, {int(np.sum(gen_ok))} gen detected)")


if __name__ == "__main__":
    main()
