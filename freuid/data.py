"""Dataset, transforms, and validation splitting for FREUID."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .augment import RecaptureSim

# Note the doubled directory layout: <root>/train/train/<id>.jpeg
_SUBDIR = {
    "train": "train/train",
    "test": "public_test/public_test",
    "sample": "train_sample/train_sample",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def img_path(data_root: str, split: str, img_id: str) -> Path:
    return Path(data_root) / _SUBDIR[split] / f"{img_id}.jpeg"


def _to_bool(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "1.0", "yes")


def load_train_df(data_root: str) -> pd.DataFrame:
    df = pd.read_csv(Path(data_root) / "train_labels.csv")
    df["is_digital"] = df["is_digital"].map(_to_bool)
    df["label"] = df["label"].astype(int)
    return df


def make_split(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Return df with a 'is_val' boolean column."""
    df = df.copy()
    if cfg.val.scheme == "group_holdout":
        holdout = set(cfg.val.holdout_types)
        if not holdout:
            raise ValueError("val.scheme=group_holdout requires val.holdout_types")
        df["is_val"] = df["type"].isin(holdout)
    elif cfg.val.scheme == "stratified":
        # stratify on (type, label) so both class and document mix are preserved
        strat = df["type"].astype(str) + "|" + df["label"].astype(str)
        skf = StratifiedKFold(n_splits=cfg.val.n_folds, shuffle=True, random_state=cfg.seed)
        df["is_val"] = False
        for i, (_, val_idx) in enumerate(skf.split(df, strat)):
            if i == cfg.val.fold:
                df.iloc[val_idx, df.columns.get_loc("is_val")] = True
                break
    else:
        raise ValueError(f"unknown val.scheme: {cfg.val.scheme}")
    return df


def build_transforms(cfg, train: bool, force_recapture: bool = False):
    """force_recapture: apply recapture sim even on the val path (recapture-robustness probe)."""
    size = cfg.data.img_size
    rc = cfg.data.get("recapture")
    if train:
        ops = [
            transforms.RandomResizedCrop(size, scale=tuple(cfg.data.rrc_scale), antialias=True),
        ]
        if cfg.data.hflip:
            ops.append(transforms.RandomHorizontalFlip())
        if rc and rc.get("prob", 0) > 0:
            ops.append(RecaptureSim(rc))           # print-and-capture artifacts
        if cfg.data.color_jitter:
            cj = cfg.data.color_jitter
            ops.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
    else:
        ops = [transforms.Resize((size, size), antialias=True)]
        if force_recapture and rc:
            probe = dict(rc); probe["prob"] = 1.0   # always-on for the diagnostic
            ops.append(RecaptureSim(probe))
    mean = tuple(cfg.data.get("mean") or IMAGENET_MEAN)   # CLIP/DINOv2 use their own norm
    std = tuple(cfg.data.get("std") or IMAGENET_STD)
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


class FraudDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str, split: str, tfm, has_label: bool):
        self.ids = df["id"].tolist()
        self.labels = df["label"].tolist() if has_label else None
        self.data_root = data_root
        self.split = split
        self.tfm = tfm
        self.has_label = has_label

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img = Image.open(img_path(self.data_root, self.split, img_id)).convert("RGB")
        x = self.tfm(img)
        if self.has_label:
            return x, torch.tensor(self.labels[i], dtype=torch.float32)
        return x, img_id


def make_loaders(df_split: pd.DataFrame, cfg):
    root = cfg.paths.data_root
    tr_df = df_split[~df_split["is_val"]].reset_index(drop=True)
    va_df = df_split[df_split["is_val"]].reset_index(drop=True)
    limit = cfg.data.get("limit")
    if limit:  # smoke / debug: subsample both splits, keep class balance
        def _balanced(df, n, seed):
            per = max(1, n // df["label"].nunique())
            picks = [g.sample(min(len(g), per), random_state=seed) for _, g in df.groupby("label")]
            return pd.concat(picks).sample(frac=1, random_state=seed).reset_index(drop=True)
        tr_df = _balanced(tr_df, limit, cfg.seed)
        va_df = _balanced(va_df, max(2, limit // 2), cfg.seed)
    tr = FraudDataset(tr_df, root, "train", build_transforms(cfg, True), has_label=True)
    va = FraudDataset(va_df, root, "train", build_transforms(cfg, False), has_label=True)
    common = dict(num_workers=cfg.data.num_workers, pin_memory=True, persistent_workers=cfg.data.num_workers > 0)
    tr_loader = DataLoader(tr, batch_size=cfg.data.batch_size, shuffle=True, drop_last=True, **common)
    va_loader = DataLoader(va, batch_size=cfg.data.batch_size, shuffle=False, **common)
    # recapture-simulated val: sensitive to print-and-capture robustness (clean val is broken, finding #0)
    va_recap_loader = None
    if cfg.data.get("recapture"):
        va_rc = FraudDataset(va_df, root, "train", build_transforms(cfg, False, force_recapture=True), has_label=True)
        va_recap_loader = DataLoader(va_rc, batch_size=cfg.data.batch_size, shuffle=False, **common)
    return tr_loader, va_loader, va_recap_loader, va_df


def list_available_test_ids(data_root: str) -> list[str]:
    d = Path(data_root) / _SUBDIR["test"]
    return sorted(p.stem for p in d.glob("*.jpeg"))


def make_test_loader(ids: list[str], cfg):
    df = pd.DataFrame({"id": ids})
    ds = FraudDataset(df, cfg.paths.data_root, "test", build_transforms(cfg, False), has_label=False)
    return DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False,
                      num_workers=cfg.data.num_workers, pin_memory=True)
