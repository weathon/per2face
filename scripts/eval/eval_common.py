"""Shared eval utilities: encoders, detection/alignment, generation.

Conventions reused from the training pipeline:
- insightface buffalo_l detection on a 300px-padded image (SCRFD misses
  tightly-cropped large faces otherwise); face.embedding is the standard
  ArcFace (w600k_r50) identity vector used as the "insightface/ArcFace" metric.
- PerFace: similarity-align 5 kps to outputs/triplet_template_224.npy -> 112.
- AVFS: FFHQ-align from 5 kps -> 160 -> resize128/centercrop112; centered on
  the celeba_hq dataset mean (outputs/avfs_embs.npz) then L2-normalized.
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
sys.path.insert(0, os.path.join(PROJ, "perface"))

from iresnet import iresnet50  # noqa: E402

PAD = 300


def get_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=os.path.join(PROJ, "weights", "insightface"),
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def detect(app, bgr):
    """Returns largest face (with .kps, .embedding) or None."""
    padded = cv2.copyMakeBorder(bgr, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    faces = app.get(padded)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    face.kps = face.kps.astype(np.float32) - PAD
    return face


class PerFaceEmb:
    def __init__(self, weights, template, device="cuda"):
        self.net = iresnet50(fp16=False)
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net.eval().to(device)
        self.template = np.load(template).astype(np.float32)
        self.device = device

    def crop(self, bgr, kps):
        M, _ = cv2.estimateAffinePartial2D(kps, self.template, method=cv2.LMEDS)
        c = cv2.warpAffine(bgr, M, (224, 224), borderValue=0)
        c = cv2.resize(c, (112, 112), interpolation=cv2.INTER_AREA)
        return c[:, :, ::-1].copy()  # BGR->RGB

    @torch.no_grad()
    def embed_crops(self, rgb_crops):
        x = torch.from_numpy(np.stack(rgb_crops)).permute(0, 3, 1, 2).float().to(self.device)
        x = (x - 127.5) / 127.5
        return F.normalize(self.net(x).float(), dim=1)


class AVFSEmb:
    def __init__(self, weights, mean_npz, device="cuda"):
        sys.path.insert(0, os.path.join(PROJ, "scripts"))
        from avfs_encoder import load_avfs_u, ffhq_align_from_kps, preprocess
        self.net = load_avfs_u(weights).to(device)
        self.align = ffhq_align_from_kps
        self.preprocess = preprocess
        self.mean = torch.from_numpy(np.load(mean_npz)["mean"]).to(device)
        self.device = device

    def crop(self, bgr, kps):
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.preprocess(self.align(pil, kps))  # tensor (3,112,112)

    @torch.no_grad()
    def embed_crops(self, tens):
        x = torch.stack(tens).to(self.device)
        e = self.net(x).float() - self.mean
        return F.normalize(e, dim=1)


def load_pipeline(ckpt, device="cuda"):
    """ckpt: 'pretrained' (original ArcFace Arc2Face) or a stepXXXXXX dir."""
    sys.path.insert(0, os.path.join(PROJ, "scripts"))
    from a2f_common import ARC2FACE, SD_BASE, encoder_from, load_tokenizer
    from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,
                           StableDiffusionPipeline, UNet2DConditionModel)
    if ckpt == "pretrained":
        enc_src, enc_sub = ARC2FACE, "encoder"
        unet = UNet2DConditionModel.from_pretrained(ARC2FACE, subfolder="arc2face")
    else:
        enc_src, enc_sub = ckpt, "encoder"
        unet = UNet2DConditionModel.from_pretrained(ckpt, subfolder="unet")
    text_encoder = encoder_from(enc_src, subfolder=enc_sub)
    tokenizer = load_tokenizer()
    vae = AutoencoderKL.from_pretrained(SD_BASE, subfolder="vae")
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(SD_BASE, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None, image_encoder=None,
        requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval(); pipe.text_encoder.eval()
    return pipe
