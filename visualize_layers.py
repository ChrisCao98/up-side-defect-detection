"""Visualize PatchCore's two feature layers (layer2 / layer3) on a few images.

Outputs, per image:
  - PCA-RGB of each layer's native feature map
  - channel-mean activation of each layer
  - layer-wise kNN anomaly map against the matching slice of the memory bank
  - the model's fused anomaly map

Usage (inside the up_patchcore-v1 image):
  python visualize_layers.py \
      --ckpt result_train_E5DJ_v2/Patchcore/F_my/v0/weights/lightning/model.ckpt \
      --out feature_viz \
      --images datasets_E5DJ/Defect/.../S61260016_D8_UP_md01.jpg \
               datasets_E5DJ/Good/E5DJ/E5DJ00052_E12_UP.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from anomalib.models import Patchcore

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 256
CMAP = "inferno"


def load_image(path: Path, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    img = Image.open(path).convert("RGB")
    disp = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    x = F.interpolate(
        x.unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    x = (x - torch.from_numpy(MEAN)[:, None, None]) / torch.from_numpy(STD)[:, None, None]
    return x.unsqueeze(0).to(device), disp


def pca_rgb(feat: torch.Tensor) -> torch.Tensor:
    """feat: [C,H,W] -> [3,H,W] in [0,1]."""
    c, h, w = feat.shape
    x = feat.reshape(c, -1).T.float()
    x = x - x.mean(0, keepdim=True)
    cov = x.T @ x / max(x.shape[0] - 1, 1)
    cov = (cov + cov.T) / 2
    evals, evecs = torch.linalg.eigh(cov)
    idx = torch.argsort(evals, descending=True)[:3]
    proj = (x @ evecs[:, idx]).reshape(h, w, 3)
    out = torch.zeros_like(proj)
    for k in range(3):
        v = proj[..., k]
        lo = torch.quantile(v.flatten(), 0.01)
        hi = torch.quantile(v.flatten(), 0.99)
        out[..., k] = ((v - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return out.permute(2, 0, 1)


def norm_map(m: torch.Tensor) -> torch.Tensor:
    lo = torch.quantile(m.flatten(), 0.01)
    hi = torch.quantile(m.flatten(), 0.99)
    return ((m - lo) / (hi - lo + 1e-8)).clamp(0, 1)


def mean_map(feat: torch.Tensor) -> torch.Tensor:
    return norm_map(feat.mean(0))


def min_dist_map(q: torch.Tensor, bank: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    out = []
    for i in range(0, q.shape[0], chunk):
        d = torch.cdist(q[i : i + chunk].float(), bank.float())
        out.append(d.min(dim=1).values)
    return torch.cat(out)


def upsample(m: torch.Tensor, size: int = IMAGE_SIZE) -> torch.Tensor:
    if m.dim() == 2:
        return F.interpolate(m[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    if m.dim() == 3:
        return F.interpolate(m[None], size=(size, size), mode="bilinear", align_corners=False)[0]
    return F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)


def upsample_nearest(m: torch.Tensor, size: int = IMAGE_SIZE) -> torch.Tensor:
    if m.dim() == 2:
        return F.interpolate(m[None, None], size=(size, size), mode="nearest")[0, 0]
    if m.dim() == 3:
        return F.interpolate(m[None], size=(size, size), mode="nearest")[0]
    return F.interpolate(m, size=(size, size), mode="nearest")


def overlay(img: Image.Image, heat: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    base = np.asarray(img, dtype=np.float32) / 255.0
    colored = plt.get_cmap(CMAP)(heat.detach().cpu().numpy())[..., :3]
    return (1 - alpha) * base + alpha * colored


def argmax_xy(m: torch.Tensor) -> tuple[int, int]:
    idx = int(torch.argmax(m).item())
    h, w = m.shape
    return idx // w, idx % w


def probe(m: torch.Tensor, xy: tuple[int, int], native: int) -> dict:
    """Sample a native map at a pixel coordinate in the 256x256 image."""
    step = IMAGE_SIZE / native
    row = min(max(int(xy[1] // step), 0), m.shape[0] - 1)
    col = min(max(int(xy[0] // step), 0), m.shape[1] - 1)
    val = float(m[row, col])
    pct = float((m < m[row, col]).float().mean() * 100)
    return {
        "xy": [int(xy[0]), int(xy[1])],
        "row_col": [row, col],
        "dist": val,
        "percentile": round(pct, 2),
    }


def annotate_max(ax, m: torch.Tensor, native: int, color: str, label: str) -> None:
    row, col = argmax_xy(m)
    step = IMAGE_SIZE / native
    x = (col + 0.5) * step
    y = (row + 0.5) * step
    ax.add_patch(plt.Circle((x - 0.5, y - 0.5), 10, fill=False, color=color, lw=1.8))
    ax.text(x + 8, y - 8, label, color=color, fontsize=8, weight="bold")


def visualize(model: Patchcore, image_path: Path, out_dir: Path, device: torch.device) -> dict:
    x, disp = load_image(image_path, device)

    with torch.no_grad():
        feats = model.model.feature_extractor(x)
        fused = model.model(x)

    pooled = {k: model.model.feature_pooler(v) for k, v in feats.items()}
    f2, f3 = pooled["layer2"].squeeze(0), pooled["layer3"].squeeze(0)  # [512,32,32], [1024,16,16]

    mem = model.model.memory_bank.float()
    mem2, mem3 = mem[:, : f2.shape[0]], mem[:, f2.shape[0] :]

    q2 = f2.permute(1, 2, 0).reshape(-1, f2.shape[0])
    q3 = f3.permute(1, 2, 0).reshape(-1, f3.shape[0])
    a2 = min_dist_map(q2, mem2).reshape(f2.shape[1], f2.shape[2])
    a3 = min_dist_map(q3, mem3).reshape(f3.shape[1], f3.shape[2])

    p2, p3 = pca_rgb(f2), pca_rgb(f3)
    m2, m3 = mean_map(f2), mean_map(f3)
    a2n, a3n = norm_map(a2), norm_map(a3)
    fused_map = fused.anomaly_map.detach().float().reshape(-1, IMAGE_SIZE, IMAGE_SIZE)[0]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    axes[0, 0].imshow(disp)
    axes[0, 0].set_title("original 256x256", fontsize=11)
    annotate_max(axes[0, 0], a2, 32, "red", "L2 max")
    annotate_max(axes[0, 0], a3, 16, "cyan", "L3 max")

    axes[0, 1].imshow(upsample_nearest(p2).permute(1, 2, 0).cpu())
    axes[0, 1].set_title(f"layer2 PCA-RGB  ({f2.shape[1]}x{f2.shape[2]}, nearest)", fontsize=11)

    axes[0, 2].imshow(upsample_nearest(p3).permute(1, 2, 0).cpu())
    axes[0, 2].set_title(f"layer3 PCA-RGB  ({f3.shape[1]}x{f3.shape[2]}, nearest)", fontsize=11)

    axes[0, 3].imshow(overlay(disp, upsample(norm_map(fused_map))))
    axes[0, 3].set_title(f"fused anomaly map  (raw score {float(fused.pred_score):.4f})", fontsize=11)

    axes[1, 0].imshow(upsample_nearest(m2).cpu(), cmap="viridis")
    axes[1, 0].set_title(f"layer2 channel-mean  ({f2.shape[1]}x{f2.shape[2]})", fontsize=11)

    axes[1, 1].imshow(upsample_nearest(m3).cpu(), cmap="viridis")
    axes[1, 1].set_title(f"layer3 channel-mean  ({f3.shape[1]}x{f3.shape[2]})", fontsize=11)

    im2 = axes[1, 2].imshow(upsample_nearest(a2n).cpu(), cmap=CMAP, vmin=0, vmax=1)
    axes[1, 2].set_title(
        f"layer2-only kNN anomaly  (max {float(a2.max()):.2f}, 512-dim)", fontsize=11
    )
    annotate_max(axes[1, 2], a2, 32, "white", "L2 max")
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.02)

    im3 = axes[1, 3].imshow(upsample_nearest(a3n).cpu(), cmap=CMAP, vmin=0, vmax=1)
    axes[1, 3].set_title(
        f"layer3-only kNN anomaly  (max {float(a3.max()):.2f}, 1024-dim)", fontsize=11
    )
    annotate_max(axes[1, 3], a3, 16, "white", "L3 max")
    fig.colorbar(im3, ax=axes[1, 3], fraction=0.046, pad=0.02)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(image_path.name, fontsize=13, weight="bold")
    out_path = out_dir / f"layers_{image_path.stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    r2, c2 = argmax_xy(a2)
    r3, c3 = argmax_xy(a3)
    xy2 = (round((c2 + 0.5) * 8), round((r2 + 0.5) * 8))
    xy3 = (round((c3 + 0.5) * 16), round((r3 + 0.5) * 16))
    fr, fc = argmax_xy(fused_map)
    summary = {
        "image": str(image_path),
        "raw_score": float(fused.pred_score),
        "layer2": {
            "shape": list(f2.shape[1:]),
            "max_dist": float(a2.max()),
            "argmax_row_col": [r2, c2],
            "argmax_pixel_xy": list(xy2),
        },
        "layer3": {
            "shape": list(f3.shape[1:]),
            "max_dist": float(a3.max()),
            "argmax_row_col": [r3, c3],
            "argmax_pixel_xy": list(xy3),
        },
        "fused": {
            "argmax_pixel_xy": [fc, fr],
            "max": float(fused_map.max()),
        },
        "probe_at_layer2_max": {
            "layer2": probe(a2, xy2, 32),
            "layer3": probe(a3, xy2, 16),
            "fused": float(fused_map[xy2[1], xy2[0]]),
        },
        "probe_at_layer3_max": {
            "layer2": probe(a2, xy3, 32),
            "layer3": probe(a3, xy3, 16),
            "fused": float(fused_map[xy3[1], xy3[0]]),
        },
        "output": str(out_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="feature_viz")
    parser.add_argument("--images", nargs="+", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Patchcore.load_from_checkpoint(args.ckpt, map_location=device).eval()

    summaries = []
    for name in args.images:
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(path)
        summary = visualize(model, path, out_dir, device)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    (out_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
