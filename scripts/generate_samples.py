"""Generate faces from PerFace embeddings with the finetuned Arc2Face model.

For each held-out input face: extract PerFace embedding (same align ->
112 -> iresnet50 path as training), generate N samples, and save a grid
with the input in the first column.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from a2f_common import (ARC2FACE, SD_BASE, build_prompt_ids, encoder_from,
                        load_tokenizer, project_face_embs_train)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "perface"))
from iresnet import iresnet50

from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,
                       StableDiffusionPipeline, UNet2DConditionModel)
from insightface.app import FaceAnalysis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoints/arc2face_perface/stepXXXXXX or 'pretrained'")
    p.add_argument("--inputs", nargs="+", required=True, help="input face image paths")
    p.add_argument("--out", default="outputs/samples")
    p.add_argument("--perface", default="checkpoints/perface/best.pth")
    p.add_argument("--template", default="outputs/triplet_template_224.npy")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    template = np.load(args.template).astype(np.float32)
    app = FaceAnalysis(name="buffalo_l", root="weights/insightface",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    net = iresnet50(fp16=False)
    net.load_state_dict(torch.load(args.perface, map_location="cpu", weights_only=True))
    net.eval().to(device)

    if args.ckpt == "pretrained":
        text_encoder = encoder_from(ARC2FACE).to(device)
        unet = UNet2DConditionModel.from_pretrained(ARC2FACE, subfolder="arc2face").to(device)
    else:
        text_encoder = encoder_from(args.ckpt).to(device)
        unet = UNet2DConditionModel.from_pretrained(args.ckpt, subfolder="unet").to(device)
    tokenizer = load_tokenizer()
    vae = AutoencoderKL.from_pretrained(SD_BASE, subfolder="vae").to(device)
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(SD_BASE, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None, image_encoder=None,
        requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    unet.eval()
    text_encoder.eval()

    cond_ids, uncond_ids, id_pos = build_prompt_ids(tokenizer, device)

    PAD = 300  # SCRFD misses tightly-cropped large faces; pad with margin
    rows = []
    for path in args.inputs:
        img = cv2.imread(path)
        padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
        faces = app.get(padded)
        if not faces:
            print(f"no face: {path}")
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        kps = face.kps.astype(np.float32) - PAD
        M, _ = cv2.estimateAffinePartial2D(kps, template, method=cv2.LMEDS)
        crop224 = cv2.warpAffine(img, M, (224, 224), borderValue=0)
        crop = cv2.resize(crop224, (112, 112), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1).float()[None].to(device)
        x = (x - 127.5) / 127.5
        with torch.no_grad():
            emb = torch.nn.functional.normalize(net(x).float(), dim=1)

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            pe = project_face_embs_train(text_encoder, emb, cond_ids, uncond_ids, id_pos)
            neg = project_face_embs_train(text_encoder, emb, cond_ids, uncond_ids, id_pos,
                                          drop_mask=torch.ones(1, dtype=torch.bool, device=device))
            images = pipe(prompt_embeds=pe.repeat(args.num_samples, 1, 1),
                          negative_prompt_embeds=neg.repeat(args.num_samples, 1, 1),
                          num_inference_steps=args.steps, guidance_scale=args.guidance,
                          generator=torch.Generator(device).manual_seed(args.seed)).images

        ref = Image.fromarray(cv2.cvtColor(crop224, cv2.COLOR_BGR2RGB)).resize((512, 512))
        row = Image.new("RGB", (512 * (1 + args.num_samples), 512))
        row.paste(ref, (0, 0))
        for k, im in enumerate(images):
            row.paste(im, (512 * (k + 1), 0))
        rows.append(row)
        print(f"generated for {path}")

    grid = Image.new("RGB", (rows[0].width, 512 * len(rows)))
    for r, row in enumerate(rows):
        grid.paste(row, (0, 512 * r))
    name = os.path.basename(args.ckpt.rstrip("/")) or "ckpt"
    out_path = os.path.join(args.out, f"grid_{name}.jpg")
    grid.save(out_path, quality=92)
    print("saved", out_path)


if __name__ == "__main__":
    main()
