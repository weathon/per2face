"""Estimate the 5-point landmark template of the SimCelebA triplet crops.

PerFace was trained on these 224x224 crops, so celeba_hq faces must be
aligned to the same convention before embedding extraction. We detect
landmarks on a sample of triplet reference images and average them, then
compare with insightface's standard arcface template scaled to 224.
"""
import glob
import os
import sys

import numpy as np

os.environ.setdefault("HF_HOME", "weights/hf")
from insightface.app import FaceAnalysis
from insightface.utils.face_align import arcface_dst

app = FaceAnalysis(name="buffalo_l", root="weights/insightface",
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(224, 224))

import cv2

paths = sorted(glob.glob("data/triplet/t*_c.jpg"))[::8][:800]
kps_all = []
for p in paths:
    img = cv2.imread(p)
    if img is None:
        continue
    faces = app.get(img)
    if len(faces) != 1:
        continue
    kps_all.append(faces[0].kps)

kps = np.stack(kps_all)
mean_kps = kps.mean(0)
print(f"detected on {len(kps_all)}/{len(paths)} images")
print("empirical mean kps (224x224):\n", np.round(mean_kps, 2))
print("std:\n", np.round(kps.std(0), 2))
print("arcface_dst scaled to 224 (mode=arcface):\n", np.round(arcface_dst * 2, 2))

np.save("outputs/triplet_template_224.npy", mean_kps)
print("saved outputs/triplet_template_224.npy")
