"""Shared pieces for Arc2Face-with-PerFace training and sampling."""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJ, "weights", "hf"))
sys.path.insert(0, os.path.join(PROJ, "Arc2Face"))

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer

from arc2face import CLIPTextModelWrapper  # noqa: E402

SD_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ARC2FACE = "FoivosPar/Arc2Face"
PROMPT = "photo of a id person"


def load_tokenizer():
    return CLIPTokenizer.from_pretrained(SD_BASE, subfolder="tokenizer")


def load_embs(npz_path):
    """Returns (files, embs) with embs L2-normalized.

    Two formats: PerFace saves 'embs' (already unit-norm); AVFS saves
    'embs_raw' + dataset 'mean' — AVFS is non-negative (positive orthant,
    raw cosines ~0.9), so we center on the dataset mean before normalizing
    to get a usable conditioning signal.
    """
    import numpy as np
    data = np.load(npz_path)
    if "embs" in data:
        return data["files"], data["embs"]
    e = data["embs_raw"] - data["mean"]
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return data["files"], e.astype(np.float32)


def encoder_from(path_or_repo, subfolder="encoder", dtype=torch.float32):
    return CLIPTextModelWrapper.from_pretrained(path_or_repo, subfolder=subfolder,
                                                torch_dtype=dtype)


def build_prompt_ids(tokenizer, device):
    """Returns (cond_ids[77], uncond_ids[77], id_token_pos)."""
    cond = tokenizer(PROMPT, truncation=True, padding="max_length",
                     max_length=tokenizer.model_max_length,
                     return_tensors="pt").input_ids[0].to(device)
    uncond = tokenizer("", truncation=True, padding="max_length",
                       max_length=tokenizer.model_max_length,
                       return_tensors="pt").input_ids[0].to(device)
    id_tok = tokenizer.encode("id", add_special_tokens=False)[0]
    pos = (cond == id_tok).nonzero()[0].item()
    return cond, uncond, pos


def project_face_embs_train(text_encoder, face_embs, cond_ids, uncond_ids,
                            id_pos, drop_mask=None):
    """Differentiable version of arc2face.utils.project_face_embs.

    face_embs: (N, 512) L2-normalized. drop_mask: (N,) bool, True = use
    unconditional empty prompt (for classifier-free guidance training).
    Returns prompt embeddings (N, 77, 768).
    """
    n = face_embs.size(0)
    hidden = text_encoder.config.hidden_size
    if drop_mask is None:
        drop_mask = torch.zeros(n, dtype=torch.bool, device=face_embs.device)
    input_ids = torch.where(drop_mask[:, None], uncond_ids[None], cond_ids[None])
    token_embs = text_encoder(input_ids=input_ids, return_token_embs=True).clone()
    padded = F.pad(face_embs, (0, hidden - face_embs.size(1)))
    keep = ~drop_mask
    token_embs[keep, id_pos] = padded[keep].to(token_embs.dtype)
    return text_encoder(input_ids=input_ids, input_token_embs=token_embs)[0]
