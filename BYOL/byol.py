"""
Bootstrap Your Own Latent (BYOL) pretraining with a ResNet-18 backbone.

RGB loading matches ``purifier/train_rgbs.py``: JPEG under ``images_path``,
top-left tile of a 3x3 grid (same as ``crop_rgb_tile(..., tile_row=0, tile_col=0)``).

Candidate names come from ``result_S10/round_49`` and ``result_spring/round_37``
``probs.csv`` rows with ``prob`` above ``min(min_score, min_score_cowls)`` from each
run's ``records.json`` (aligned with ``purifier/train_embeddings.py`` paths), not from
enumerating every file under the image directory.

After training, by default the script saves ``byol_loss_history.json`` /
``byol_loss_history.png``, ``byol_embeddings.npz`` (names, ``is_sl``,
512-D **target**-encoder vectors) and ``byol_umap_sl_other_seeds.png`` (UMAP,
three panels with ``random_state`` 42, 123, 2026).
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
import json


def crop_rgb_tile(
    img: Image.Image, grid_size: int = 3, tile_row: int = 0, tile_col: int = 0
) -> Image.Image:
    width, height = img.size
    tile_w = width // grid_size
    tile_h = height // grid_size
    left = tile_col * tile_w
    upper = tile_row * tile_h
    right = left + tile_w
    lower = upper + tile_h
    return img.crop((left, upper, right, lower))


def filter_existing_names(names: Sequence[str], images_path: str) -> List[str]:
    out: List[str] = []
    for n in names:
        s = str(n)
        if os.path.exists(os.path.join(images_path, f"{s}.jpg")):
            out.append(s)
    if len(out) == 0:
        raise RuntimeError("No valid image files found. Check images_path and names.")
    return out


def byol_train_transform(image_size: int = 224) -> transforms.Compose:
    """Stochastic augmentations; call twice per image for two views."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # Astronomical targets are isotropic - random rotations are physically natural
            transforms.RandomRotation(degrees=(0, 360)),
            # Reduced color jitter to prevent disrupting physical filter ratios
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.3
            ),
            # Grayscale is removed completely to preserve crucial red-foreground / blue-background contrast
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic preprocessing (matches ``purifier/train_rgbs.py`` eval style)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def resolve_out_dir_and_checkpoint(save_path: str) -> Tuple[str, str]:
    """If ``save_path`` ends with ``.pt``, it is the checkpoint file; otherwise it is an output directory."""
    save_path = os.path.abspath(save_path)
    if save_path.endswith(".pt"):
        out_dir = os.path.dirname(save_path) or "."
        ckpt = save_path
    else:
        out_dir = save_path
        os.makedirs(out_dir, exist_ok=True)
        ckpt = os.path.join(out_dir, "byol_resnet18.pt")
    return out_dir, ckpt


def build_sl_other_embedding_manifest(
    sl_names: List[str], other_names: List[str], images_path: str
) -> Tuple[List[str], np.ndarray]:
    """Ordered names and ``is_sl`` flags (1 = SL, 0 = other), images filtered to existing JPEGs."""
    sl_keep = filter_existing_names(sl_names, images_path)
    sl_set = set(sl_keep)
    other_keep = [
        n
        for n in filter_existing_names(other_names, images_path)
        if n not in sl_set
    ]
    names = sl_keep + other_keep
    is_sl = np.concatenate(
        [
            np.ones(len(sl_keep), dtype=np.int8),
            np.zeros(len(other_keep), dtype=np.int8),
        ]
    )
    return names, is_sl


class EmbeddingImageDataset(Dataset):
    """Single deterministic view per name (for encoder embeddings)."""

    def __init__(self, names: List[str], images_path: str, transform):
        self.names = list(names)
        self.images_path = images_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        name = self.names[idx]
        path = os.path.join(self.images_path, f"{name}.jpg")
        img = Image.open(path).convert("RGB")
        rgb_tile = crop_rgb_tile(img, grid_size=3, tile_row=0, tile_col=0)
        return self.transform(rgb_tile)


def plot_umap_multipanel(
    embeddings: np.ndarray,
    is_sl: np.ndarray,
    out_fig_path: str,
    seeds: Tuple[int, int, int] = (42, 123, 2026),
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """2-D UMAP with umap-learn, one panel per random seed; SL points highlighted."""
    try:
        import umap  # type: ignore
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for this visualization. Install with: pip install umap-learn"
        ) from e

    Z = np.ascontiguousarray(embeddings.astype(np.float32, copy=False))
    n_samples = int(Z.shape[0])
    if n_samples < 3:
        raise ValueError(f"UMAP needs at least 3 samples; got {n_samples}.")
    nn = min(max(2, n_neighbors), n_samples - 1)

    sl_m = is_sl.astype(bool)
    oth_m = ~sl_m

    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 5), constrained_layout=True)
    if len(seeds) == 1:
        axes = [axes]
    for ax, seed in zip(axes, seeds):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=min_dist,
            metric='cosine',
            random_state=seed,
            n_jobs=-1,
        )
        emb2d = reducer.fit_transform(Z)
        ax.scatter(
            emb2d[oth_m, 0],
            emb2d[oth_m, 1],
            s=8,
            c="0.55",
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            label="other",
        )
        ax.scatter(
            emb2d[sl_m, 0],
            emb2d[sl_m, 1],
            s=24,
            c="crimson",
            alpha=0.9,
            linewidths=0.35,
            edgecolors="darkred",
            label="SL",
        )
        ax.set_title(f"UMAP, random_state={seed}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(loc="best", fontsize=9)
    fig_dir = os.path.dirname(os.path.abspath(out_fig_path))
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(out_fig_path, dpi=160)
    plt.close(fig)


class BYOLImageDataset(Dataset):
    """One sample = two augmented views of the same RGB tile."""

    def __init__(self, names: List[str], images_path: str, transform):
        self.names = list(names)
        self.images_path = images_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx]
        path = os.path.join(self.images_path, f"{name}.jpg")
        img = Image.open(path).convert("RGB")
        rgb_tile = crop_rgb_tile(img, grid_size=3, tile_row=0, tile_col=0)
        v1 = self.transform(rgb_tile)
        v2 = self.transform(rgb_tile)
        return v1, v2


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_resnet18_encoder(device: torch.device) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    return m.to(device)


def regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """BYOL loss: MSE on L2-normalized representations (per-sample mean over dim)."""
    p = F.normalize(p, dim=-1, p=2, eps=1e-8)
    z = F.normalize(z, dim=-1, p=2, eps=1e-8)
    return ((p - z) ** 2).sum(dim=-1).mean()


@torch.no_grad()
def _ema_update(
    online: nn.Module, target: nn.Module, momentum: float
) -> None:
    for op, tp in zip(online.parameters(), target.parameters()):
        tp.data = tp.data * momentum + op.data * (1.0 - momentum)


class BYOL(nn.Module):
    """
    Online branch: encoder -> projector -> predictor.
    Target branch: encoder -> projector (EMA of online encoder + projector).
    """

    def __init__(
        self,
        proj_hidden: int = 2048,
        proj_out: int = 256,
        pred_hidden: int = 512,
        device: torch.device | None = None,
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        self.online_encoder = build_resnet18_encoder(dev)
        emb = 512
        self.online_projector = MLP(emb, proj_hidden, proj_out).to(dev)
        self.online_predictor = MLP(proj_out, pred_hidden, proj_out).to(dev)

        self.target_encoder = build_resnet18_encoder(dev)
        self.target_projector = MLP(emb, proj_hidden, proj_out).to(dev)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self._init_target_from_online()

    @torch.no_grad()
    def _init_target_from_online(self) -> None:
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        z1_o = self.online_projector(self.online_encoder(v1))
        z2_o = self.online_projector(self.online_encoder(v2))
        z1_t = self.target_projector(self.target_encoder(v1))
        z2_t = self.target_projector(self.target_encoder(v2))

        p1 = self.online_predictor(z1_o)
        p2 = self.online_predictor(z2_o)

        loss = regression_loss(p1, z2_t) + regression_loss(p2, z1_t)
        return loss

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        _ema_update(self.online_encoder, self.target_encoder, momentum)
        _ema_update(self.online_projector, self.target_projector, momentum)


@torch.no_grad()
def extract_target_encoder_embeddings(
    model: BYOL,
    names: List[str],
    images_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """512-D embeddings from the BYOL **target** encoder (eval mode)."""
    model.eval()
    tfm = eval_transform(224)
    ds = EmbeddingImageDataset(names, images_path, tfm)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    enc = model.target_encoder
    parts: List[np.ndarray] = []
    for batch in tqdm(loader, desc="target_encoder embeddings"):
        batch = batch.to(device, non_blocking=True)
        z = enc(batch)
        parts.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(parts, axis=0)


def plot_byol_loss_history(
    history: dict,
    save_path: str,
    warmup_epochs: int = 0,
) -> None:
    """Plot epoch-mean and per-batch BYOL loss curves."""
    epoch_losses = history["epoch_loss"]
    batch_losses = history.get("batch_loss", [])
    n_epochs = len(epoch_losses)
    epochs = np.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_epoch = axes[0]
    ax_epoch.plot(epochs, epoch_losses, "o-", color="steelblue", linewidth=2, markersize=5)
    if warmup_epochs > 0 and warmup_epochs < n_epochs:
        ax_epoch.axvline(
            warmup_epochs + 0.5,
            color="gray",
            linestyle="--",
            alpha=0.75,
            label="EMA warmup end",
        )
    ax_epoch.set_xlabel("Epoch")
    ax_epoch.set_ylabel("Mean BYOL loss")
    ax_epoch.set_title("Epoch mean loss")
    ax_epoch.grid(True, alpha=0.3)
    if warmup_epochs > 0:
        ax_epoch.legend(fontsize=8)

    ax_batch = axes[1]
    if batch_losses:
        steps = np.arange(1, len(batch_losses) + 1)
        ax_batch.plot(steps, batch_losses, color="steelblue", alpha=0.4, linewidth=0.8)
        epoch_x = history.get("epoch_batch_end") or list(range(1, n_epochs + 1))
        ax_batch.plot(
            epoch_x,
            epoch_losses,
            "o-",
            color="darkred",
            linewidth=1.5,
            markersize=4,
            label="Epoch mean",
        )
        ax_batch.legend(fontsize=8)
    ax_batch.set_xlabel("Batch step")
    ax_batch.set_ylabel("BYOL loss")
    ax_batch.set_title("Per-batch loss")
    ax_batch.grid(True, alpha=0.3)

    fig.suptitle("BYOL training loss history", y=1.02)
    fig.tight_layout()
    fig_dir = os.path.dirname(os.path.abspath(save_path))
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss history figure to {save_path}")


def train_byol(
    model: BYOL,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    momentum_target: float,
    warmup_epochs: int = 0,
    momentum_min: float = 0.99,
    history: dict | None = None,
) -> dict:
    params = (
        list(model.online_encoder.parameters())
        + list(model.online_projector.parameters())
        + list(model.online_predictor.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    if history is None:
        history = {"epoch_loss": [], "batch_loss": [], "epoch_batch_end": []}
    else:
        # Ensure all key lists exist
        for key in ["epoch_loss", "batch_loss", "epoch_batch_end"]:
            if key not in history:
                history[key] = []
                
    epoch_offset = len(history["epoch_loss"])

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        pbar = tqdm(loader, desc=f"BYOL epoch {epoch_offset + epoch + 1}/{epoch_offset + epochs}")
        for v1, v2 in pbar:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            
            global_epoch = epoch_offset + epoch
            if warmup_epochs > 0 and global_epoch < warmup_epochs:
                t = (global_epoch + 1) / float(warmup_epochs)
                m = momentum_min + (momentum_target - momentum_min) * t
            else:
                m = momentum_target

            optimizer.zero_grad(set_to_none=True)
            loss = model(v1, v2)
            loss.backward()
            optimizer.step()
            model.update_target(momentum=m)

            bs = v1.size(0)
            loss_val = float(loss.item())
            history["batch_loss"].append(loss_val)
            epoch_loss += loss_val * bs
            n += bs
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        avg = epoch_loss / max(n, 1)
        history["epoch_loss"].append(avg)
        history["epoch_batch_end"].append(len(history["batch_loss"]))
        tqdm.write(f"epoch {epoch_offset + epoch + 1}: mean loss = {avg:.6f}")

    return history


def _prob_threshold_from_records(record_path: str) -> float:
    """Same rule as ``purifier/train_rgbs.candidate_df`` / ``train_embeddings`` detector candidates."""
    with open(record_path, "r") as f:
        records = json.load(f)
    return float(min(records["min_score"], records["min_score_cowls"]))

def load_sl_and_other_names(count_stats):
    with open(count_stats, 'r') as f:
        stats = json.load(f)
        
    sl_names = stats['sl_names']
    other_names = stats['unlabeled_names']
    
    return sl_names, other_names

# def load_sl_and_other_names(repo_root: str, dataframe_path: str) -> Tuple[List[str], List[str]]:
#     """
#     SL and non-SL names from WEB (S10 round 49) and SPRING (round 37) ``probs.csv``,
#     restricted to rows with ``prob`` above the minimum score in each run's ``records.json``
#     (same paths as ``purifier/train_embeddings.py``). Only these names are used for BYOL
#     and embeddings; JPEGs must still exist under ``images_path`` (``filter_existing_names``).
#     """
#     web_probs_path = os.path.join(repo_root, "result_S10_exp_3", "round_51", "probs.csv")
#     web_record_path = os.path.join(repo_root, "result_S10_exp_3", "round_51", "records.json")
#     spring_probs_path = os.path.join(repo_root, "result_spring_exp_2", "round_42", "probs.csv")
#     spring_record_path = os.path.join(repo_root, "result_spring_exp_2", "round_42", "records.json")

#     web_probs = pd.read_csv(web_probs_path, low_memory=False)
#     web_threshold = web_probs[web_probs['selected_sl'] == 1]['prob'].min()
#     web_candidates = web_probs[web_probs["prob"] > web_threshold]
    
#     spring_probs = pd.read_csv(spring_probs_path, low_memory=False)
#     spring_threshold = spring_probs[spring_probs['selected_sl'] == 1]['prob'].min()
#     spring_candidates = spring_probs[spring_probs["prob"] > spring_threshold]
    
#     web_sl_names = web_candidates[web_candidates["selected_sl"] == 1]["name"].astype(str).tolist()
#     web_other_names = web_candidates[web_candidates["selected_sl"] == 0]["name"].astype(str).tolist()
    
#     spring_sl_names = spring_candidates[spring_candidates["selected_sl"] == 1]["name"].astype(str).tolist()
#     spring_other_names = spring_candidates[spring_candidates["selected_sl"] == 0]["name"].astype(str).tolist()
    
#     sl_names = list(dict.fromkeys(web_sl_names + spring_sl_names))
#     other_names = list(dict.fromkeys(web_other_names + spring_other_names))
    
#     return sl_names, other_names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BYOL pretraining (ResNet-18)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--momentum-target", type=float, default=0.996)
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Linear EMA ramp from --momentum-min to --momentum-target over this many epochs",
    )
    p.add_argument(
        "--momentum-min",
        type=float,
        default=0.99,
        help="Starting target-network EMA momentum (used until warmup ends)",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--save-path",
        type=str,
        default="cl_results",
        help="Output directory (default), or a path ending in .pt for the checkpoint file only.",
    )
    p.add_argument(
        "--sl-only",
        action="store_true",
        help="Train only on SL (positive) image names; default uses SL + other.",
    )
    p.add_argument(
        "--skip-embed-viz",
        action="store_true",
        help="Skip saving embeddings + UMAP figure after training.",
    )
    p.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors for the post-train visualization.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint if it exists",
    )
    return p.parse_args()


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
    images_path = os.path.abspath(
        os.path.join(_script_dir, "..", "..", "updated_datasets", "JWST_images")
    )
    dataframe_path = os.path.abspath(
        os.path.join(_script_dir, "..", "..", "updated_datasets", "JWST_SL_discovery_catalog_with_embedding_index.csv")
    )
    
    args = parse_args()
    out_dir, ckpt_file = resolve_out_dir_and_checkpoint(args.save_path)

    sl_names, other_names = load_sl_and_other_names('count_stats.json')
    
    print(f"Number of SL names: {len(sl_names)}")
    print(f"Number of other names: {len(other_names)}")

    if args.sl_only:
        train_names = filter_existing_names(sl_names, images_path)
        print(f"Training on SL-only images: {len(train_names)}")
    else:
        total = list(dict.fromkeys(sl_names + other_names))
        train_names = filter_existing_names(total, images_path)
        print(
            f"Training on SL + other: {len(train_names)} "
            f"(sl list {len(sl_names)}, other list {len(other_names)})"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    transform = byol_train_transform(224)
    dataset = BYOLImageDataset(train_names, images_path, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = BYOL(device=device).to(device)
    
    loss_history = {"epoch_loss": [], "batch_loss": [], "epoch_batch_end": []}
    loss_json_path = os.path.join(out_dir, "byol_loss_history.json")
    
    if args.resume and os.path.exists(ckpt_file):
        print(f"Loading checkpoint from {ckpt_file}...")
        checkpoint = torch.load(ckpt_file, map_location=device)
        model.online_encoder.load_state_dict(checkpoint["online_encoder"])
        model.online_projector.load_state_dict(checkpoint["online_projector"])
        model.online_predictor.load_state_dict(checkpoint["online_predictor"])
        model.target_encoder.load_state_dict(checkpoint["target_encoder"])
        model.target_projector.load_state_dict(checkpoint["target_projector"])
        print("Resumed model weights successfully.")
        
        if os.path.exists(loss_json_path):
            try:
                with open(loss_json_path, "r") as f:
                    loss_history = json.load(f)
                print(f"Loaded existing loss history ({len(loss_history['epoch_loss'])} epochs completed).")
            except Exception as e:
                print(f"Warning: Failed to load loss history from {loss_json_path}: {e}")

    loss_history = train_byol(
        model,
        loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum_target=args.momentum_target,
        warmup_epochs=args.warmup_epochs,
        momentum_min=args.momentum_min,
        history=loss_history,
    )

    with open(loss_json_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    print(f"Saved loss history to {loss_json_path}")

    loss_fig_path = os.path.join(out_dir, "byol_loss_history.png")
    plot_byol_loss_history(
        loss_history,
        loss_fig_path,
        warmup_epochs=args.warmup_epochs,
    )

    payload = {
        "online_encoder": model.online_encoder.state_dict(),
        "online_projector": model.online_projector.state_dict(),
        "online_predictor": model.online_predictor.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "target_projector": model.target_projector.state_dict(),
        "train_names_count": len(train_names),
        "images_path": images_path,
        "loss_history": loss_history,
    }
    torch.save(payload, ckpt_file)
    print(f"Saved checkpoint to {ckpt_file}")

    if not args.skip_embed_viz:
        emb_names, is_sl = build_sl_other_embedding_manifest(
            sl_names, other_names, images_path
        )
        embeddings = extract_target_encoder_embeddings(
            model,
            emb_names,
            images_path,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        emb_npz = os.path.join(out_dir, "byol_embeddings.npz")
        np.savez(
            emb_npz,
            name=np.array(emb_names, dtype=object),
            is_sl=is_sl,
            embedding=embeddings,
        )
        print(f"Saved embeddings ({embeddings.shape[0]} x {embeddings.shape[1]}) to {emb_npz}")
        fig_path = os.path.join(out_dir, "byol_umap_sl_other_seeds.png")
        plot_umap_multipanel(
            embeddings,
            is_sl,
            fig_path,
            seeds=(42, 123, 2026),
            n_neighbors=args.umap_n_neighbors,
        )
        print(f"Saved UMAP figure to {fig_path}")
