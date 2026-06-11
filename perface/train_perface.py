"""Finetune ArcFace (iresnet50, MS1MV3) into PerFace with triplet loss.

Paper recipe (Sec 4.1): SGD lr=0.01, momentum=0.9, weight decay=5e-4,
batch=32, margin=0.1, loss Eq.(1):
    L = max(0, cos(x, x-) - cos(x, x+) + m)
Trained on D2 (consistent annotations only); eval on consistent triplets:
correct iff cos(ref, pos) > cos(ref, neg).
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import TripletDataset, load_annotations, make_splits
from iresnet import iresnet50


def evaluate(net, loader, device):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for ref, xp, xn in loader:
            n = ref.size(0)
            x = torch.cat([ref, xp, xn]).to(device, non_blocking=True)
            e = F.normalize(net(x).float(), dim=1)
            er, ep, en = e[:n], e[n:2 * n], e[2 * n:]
            correct += ((er * ep).sum(1) > (er * en).sum(1)).sum().item()
            total += n
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/triplet")
    p.add_argument("--csv", default="data/triplet_answers.csv")
    p.add_argument("--weights", default="weights/ms1mv3_arcface_r50_fp16.pth")
    p.add_argument("--out", default="checkpoints/perface")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-d1", action="store_true", help="train on all triplets, not just consistent")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    items = load_annotations(args.csv, root=args.data_root)
    splits = make_splits(items, seed=args.seed)
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump(splits, f)

    train_items = [items[i] for i in splits["train"]
                   if args.use_d1 or items[i]["consistent"]]
    val_items = [items[i] for i in splits["val"] if items[i]["consistent"]]
    test_items = [items[i] for i in splits["test"] if items[i]["consistent"]]
    print(f"train={len(train_items)} ({'D1' if args.use_d1 else 'D2'}), "
          f"val={len(val_items)}, test={len(test_items)} (val/test consistent only)")

    dl_kw = dict(batch_size=args.batch_size, num_workers=8, pin_memory=True)
    train_loader = DataLoader(TripletDataset(args.data_root, train_items, train=True),
                              shuffle=True, drop_last=True, **dl_kw)
    val_loader = DataLoader(TripletDataset(args.data_root, val_items), **dl_kw)
    test_loader = DataLoader(TripletDataset(args.data_root, test_items), **dl_kw)

    net = iresnet50(fp16=False)
    net.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    net.to(device)

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    print(f"pretrained: val acc={evaluate(net, val_loader, device):.4f}, "
          f"test acc={evaluate(net, test_loader, device):.4f}")

    best_val, best_epoch = 0.0, -1
    for epoch in range(args.epochs):
        net.train()
        running, steps = 0.0, 0
        for ref, xp, xn in train_loader:
            n = ref.size(0)
            x = torch.cat([ref, xp, xn]).to(device, non_blocking=True)
            e = F.normalize(net(x).float(), dim=1)
            er, ep, en = e[:n], e[n:2 * n], e[2 * n:]
            loss = F.relu((er * en).sum(1) - (er * ep).sum(1) + args.margin).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            steps += 1
        val_acc = evaluate(net, val_loader, device)
        print(f"epoch {epoch:3d}  loss={running / steps:.4f}  val_acc={val_acc:.4f}", flush=True)
        torch.save(net.state_dict(), os.path.join(args.out, "last.pth"))
        if epoch % 5 == 4:
            torch.save(net.state_dict(), os.path.join(args.out, f"epoch{epoch:03d}.pth"))
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save(net.state_dict(), os.path.join(args.out, "best.pth"))

    net.load_state_dict(torch.load(os.path.join(args.out, "best.pth"), weights_only=True))
    test_acc = evaluate(net, test_loader, device)
    print(f"best epoch {best_epoch} (val {best_val:.4f}) -> TEST acc = {test_acc:.4f}")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val": best_val, "test_acc": test_acc}, f)


if __name__ == "__main__":
    main()
