"""Download Arc2Face + SD1.5 base checkpoints into ./weights/hf.

Sources (logged per task.md):
- hf.co/FoivosPar/Arc2Face: finetuned UNet (arc2face/), finetuned CLIP text
  encoder (encoder/), and antelopev2 ArcFace onnx (arcface.onnx)
- hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5: VAE/tokenizer/scheduler
"""
import os

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "..", "weights", "hf"))
from huggingface_hub import snapshot_download

p = snapshot_download(
    "FoivosPar/Arc2Face",
    allow_patterns=["arc2face/*", "encoder/*", "antelopev2/*"],
)
print("Arc2Face:", p)

p = snapshot_download(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    allow_patterns=[
        "model_index.json", "vae/*", "tokenizer/*", "scheduler/*",
        "text_encoder/config.json", "feature_extractor/*", "unet/config.json",
    ],
    ignore_patterns=["*.ckpt", "*.bin"] ,
)
print("SD1.5:", p)
