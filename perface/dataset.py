"""SimCelebA triplet dataset (PerFace, Kumagai et al., ICIP 2025).

Each triplet t####: c = reference image, a/b = same target identity
face-swapped with two other sources. triplet_answers.csv holds three
annotator answers ('a' or 'b') for "which is more similar to c".
Majority answer -> positive (x+), the other -> negative (x-).
D2 = triplets where all three annotators agree.
"""
import csv
import os
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_annotations(csv_path, root=None):
    """If root is given, drop triplets with missing/empty images
    (t4963_c.jpg is 0 bytes in the released tarball)."""
    items = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            tid = row["tripletId"]
            if root is not None:
                paths = [os.path.join(root, f"{tid}_{s}.jpg") for s in "abc"]
                if any(not os.path.isfile(p) or os.path.getsize(p) == 0 for p in paths):
                    continue
            answers = [row["ans1"], row["ans2"], row["ans3"]]
            majority = Counter(answers).most_common(1)[0][0]
            items.append({
                "tid": tid,
                "majority": majority,
                "consistent": len(set(answers)) == 1,
            })
    return items


def make_splits(items, seed=42, val_frac=0.1, test_frac=0.1):
    """Random triplet-level split (matches paper eval setting [ii])."""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    n_test = int(len(idx) * test_frac)
    return {
        "val": sorted(idx[:n_val]),
        "test": sorted(idx[n_val:n_val + n_test]),
        "train": sorted(idx[n_val + n_test:]),
    }


class TripletDataset(Dataset):
    def __init__(self, root, items, train=False, size=112):
        self.root = root
        self.items = items
        self.train = train
        self.size = size

    def __len__(self):
        return len(self.items)

    def _load(self, path, flip):
        im = Image.open(path).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float()
        return (x - 127.5) / 127.5

    def __getitem__(self, i):
        it = self.items[i]
        pos, neg = (it["majority"], "b" if it["majority"] == "a" else "a")
        flip = self.train and random.random() < 0.5
        ref = self._load(os.path.join(self.root, f"{it['tid']}_c.jpg"), flip)
        xp = self._load(os.path.join(self.root, f"{it['tid']}_{pos}.jpg"), flip)
        xn = self._load(os.path.join(self.root, f"{it['tid']}_{neg}.jpg"), flip)
        return ref, xp, xn
