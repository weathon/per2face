"""AVFS encoder wrapper (Sony Research, 'A View From Somewhere').

Loads avfs_u (unconditional/group-level variant): ResNet18 -> 128-dim fc,
then row-selection of the 22 informative dims per the official registry.
Source: github.com/SonyResearch/a_view_from_somewhere,
checkpoint https://zenodo.org/record/7878655/files/avfs_u.pth

Expected input (per their eval loader): FFHQ-aligned 160x160 face ->
resize 128 -> center crop 112 -> (x-0.5)/0.5.
"""
import os
import sys

import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "a_view_from_somewhere"))

from avfs.modeling.face_encoder import resnet18  # noqa: E402

# avfs_model_registry["avfs_u"]["dims"] from avfs/build_avfs.py
AVFS_U_DIMS = [70, 68, 34, 72, 95, 66, 124, 44, 40, 73, 107, 17, 75, 19, 29,
               97, 30, 91, 3, 112, 84, 1]


def load_avfs_u(weights=os.path.join(PROJ, "weights", "avfs_u.pth")):
    model = resnet18(num_output_dims=128)
    sd = torch.load(weights, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=True)
    model.fc.weight = torch.nn.Parameter(model.fc.weight[AVFS_U_DIMS, :])
    return model.eval()


def ffhq_align_from_kps(img_pil, kps, output_size=160, transform_size=640):
    """FFHQ alignment driven by insightface 5-point kps.

    Arc2Face's image_align (the FFHQ recipe) only uses the eye-cluster means
    and outer mouth corners of the 68-point layout, so we synthesize a
    68-point array carrying our 5 kps at the right indices.
    """
    sys.path.insert(0, os.path.join(PROJ, "Arc2Face"))
    from arc2face.utils import image_align
    lm = np.zeros((68, 2), dtype=np.float32)
    lm[36:42] = kps[0]  # left eye cluster -> mean = kps[0]
    lm[42:48] = kps[1]  # right eye cluster
    lm[48] = kps[3]     # mouth left corner
    lm[54] = kps[4]     # mouth right corner
    return image_align(img_pil, lm, output_size=output_size,
                       transform_size=transform_size, enable_padding=True)


def preprocess(img_pil_160):
    """160x160 aligned PIL -> normalized (3,112,112) tensor."""
    im = img_pil_160.resize((128, 128))
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = a[8:120, 8:120]  # center crop 112
    x = torch.from_numpy(a).permute(2, 0, 1)
    return (x - 0.5) / 0.5
