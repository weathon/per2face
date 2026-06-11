"""Finetune Arc2Face (UNet + text encoder) with PerFace conditioning.

Follows the Arc2Face recipe: SD1.5 backbone, pseudo-prompt "photo of a id
person" whose <id> token embedding is replaced by the (zero-padded) face
embedding; both text encoder and UNet are optimized with the standard
noise-prediction loss. Conditioning embeddings come from the frozen
PerFace encoder (precomputed by extract_embeddings.py).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from a2f_common import (ARC2FACE, SD_BASE, build_prompt_ids, encoder_from,
                        load_tokenizer, project_face_embs_train)

from diffusers import (AutoencoderKL, DDPMScheduler,
                       DPMSolverMultistepScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)


class FaceEmbDataset(Dataset):
    def __init__(self, image_dir, files, embs, size=512):
        self.image_dir = image_dir
        self.files = files
        self.embs = embs
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.image_dir, self.files[i])).convert("RGB")
        im = im.resize((self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 127.5 - 1.0
        return x, torch.from_numpy(self.embs[i])


@torch.no_grad()
def sample_grid(pipe, embs, cond_ids, uncond_ids, id_pos, path, seed=1234):
    pipe.unet.eval()
    pipe.text_encoder.eval()
    with torch.autocast("cuda", torch.bfloat16):
        prompt_embeds = project_face_embs_train(
            pipe.text_encoder, embs, cond_ids, uncond_ids, id_pos)
        neg = project_face_embs_train(
            pipe.text_encoder, embs, cond_ids, uncond_ids, id_pos,
            drop_mask=torch.ones(len(embs), dtype=torch.bool, device=embs.device))
        images = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=neg,
                      num_inference_steps=25, guidance_scale=3.0,
                      generator=torch.Generator("cuda").manual_seed(seed)).images
    w, h = images[0].size
    grid = Image.new("RGB", (w * len(images), h))
    for k, im in enumerate(images):
        grid.paste(im, (k * w, 0))
    grid.save(path)
    pipe.unet.train()
    pipe.text_encoder.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="/home/wg25r/fastdata/marshall/celeba_hq_30k/originals")
    p.add_argument("--embs", default="outputs/perface_embs.npz")
    p.add_argument("--out", default="checkpoints/arc2face_perface")
    p.add_argument("--holdout", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--cond-dropout", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    data = np.load(args.embs)
    files, embs = data["files"], data["embs"]
    order = np.argsort(files)
    files, embs = files[order], embs[order]
    train_files, train_embs = files[:-args.holdout], embs[:-args.holdout]
    hold_files, hold_embs = files[-args.holdout:], embs[-args.holdout:]
    with open(os.path.join(args.out, "holdout.json"), "w") as f:
        json.dump([str(x) for x in hold_files], f)
    print(f"train={len(train_files)}, holdout={len(hold_files)}")

    tokenizer = load_tokenizer()
    text_encoder = encoder_from(ARC2FACE).to(device)
    unet = UNet2DConditionModel.from_pretrained(ARC2FACE, subfolder="arc2face").to(device)
    vae = AutoencoderKL.from_pretrained(SD_BASE, subfolder="vae").to(device)
    vae.requires_grad_(False)
    noise_sched = DDPMScheduler.from_pretrained(SD_BASE, subfolder="scheduler")

    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    unet.train()
    text_encoder.train()

    cond_ids, uncond_ids, id_pos = build_prompt_ids(tokenizer, device)

    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(SD_BASE, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None, image_encoder=None,
        requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)

    sample_embs = torch.from_numpy(hold_embs[:6].copy()).to(device)

    opt = torch.optim.AdamW(list(unet.parameters()) + list(text_encoder.parameters()),
                            lr=args.lr, weight_decay=1e-2)

    ds = FaceEmbDataset(args.images, train_files, train_embs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True, persistent_workers=True)

    def lr_lambda(step):
        return min(1.0, step / max(1, args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    sample_grid(pipe, sample_embs, cond_ids, uncond_ids, id_pos,
                os.path.join(args.out, "samples_step000000.jpg"))
    print("baseline sample saved (pretrained Arc2Face, PerFace embeddings)", flush=True)

    step, ema_loss = 0, None
    while step < args.steps:
        for x, emb in dl:
            if step >= args.steps:
                break
            x = x.to(device, non_blocking=True)
            emb = emb.to(device, non_blocking=True)
            n = x.size(0)

            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                latents = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.float()
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (n,), device=device)
            noisy = noise_sched.add_noise(latents, noise, t)

            drop = torch.rand(n, device=device) < args.cond_dropout
            with torch.autocast("cuda", torch.bfloat16):
                prompt_embeds = project_face_embs_train(
                    text_encoder, emb, cond_ids, uncond_ids, id_pos, drop_mask=drop)
                pred = unet(noisy, t, encoder_hidden_states=prompt_embeds).sample
                loss = F.mse_loss(pred.float(), noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(unet.parameters()) + list(text_encoder.parameters()), 1.0)
            opt.step()
            sched.step()

            ema_loss = loss.item() if ema_loss is None else 0.99 * ema_loss + 0.01 * loss.item()
            step += 1
            if step % 100 == 0:
                print(f"step {step:6d}  loss={loss.item():.4f}  ema={ema_loss:.4f}", flush=True)
            if step % args.sample_every == 0:
                sample_grid(pipe, sample_embs, cond_ids, uncond_ids, id_pos,
                            os.path.join(args.out, f"samples_step{step:06d}.jpg"))
            if step % args.save_every == 0 or step == args.steps:
                ckpt = os.path.join(args.out, f"step{step:06d}")
                unet.save_pretrained(os.path.join(ckpt, "unet"))
                text_encoder.save_pretrained(os.path.join(ckpt, "encoder"))
                print(f"saved {ckpt}", flush=True)

    print("done")


if __name__ == "__main__":
    main()
