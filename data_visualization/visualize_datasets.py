#!/usr/bin/env python
import argparse
import os
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Default dataset paths from training/train_physics.py L767-768
DEFAULT_UX_PATH = "/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251028_150110/UX_nan_filtered.npy"
DEFAULT_UY_PATH = "/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251028_150110/UY_nan_filtered.npy"


def nan_sanitize(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_array(path):
    if path is None:
        return None
    arr = np.load(path)
    arr = nan_sanitize(arr)
    # unify shape to [T, nx, ny]
    if arr.ndim == 4:  # [N, T, nx, ny]
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported array shape for {path}: {arr.shape}")
    return arr.astype(np.float32)


def load_mask(path, target_shape):
    if path is None:
        return None
    m = np.load(path)
    m = nan_sanitize(m)
    # squeeze to 2D [nx, ny]
    m = np.squeeze(m)
    if m.ndim == 3:  # [C, nx, ny] -> take first
        m = m[0]
    if m.ndim != 2:
        # try reshape from [1,1,nx,ny]
        try:
            m = m.reshape(target_shape)
        except Exception:
            raise ValueError(f"Unsupported mask shape: {m.shape}, expected 2D {target_shape}")
    if m.shape != target_shape:
        raise ValueError(f"Mask shape mismatch: got {m.shape}, expected {target_shape}")
    return m.astype(np.float32)


def percentiles_vmin_vmax(frame, p1=1.0, p99=99.0):
    vals = frame[np.isfinite(frame)]
    vals = vals[np.abs(vals) > 1e-8]
    if vals.size < 50:
        vals = frame.reshape(-1)
    if vals.size == 0:
        vmin = float(np.nanmin(frame))
        vmax = float(np.nanmax(frame))
    else:
        vmin = float(np.percentile(vals, p1))
        vmax = float(np.percentile(vals, p99))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            m = float(vals.mean())
            s = float(vals.std())
            if s > 0:
                vmin = m - 3.0 * s
                vmax = m + 3.0 * s
            else:
                vmin = float(vals.min())
                vmax = float(vals.max())
    # small guard
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax == vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def save_frame_image(frame, out_path, cmap="turbo", vmin=None, vmax=None, title=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    if vmin is None or vmax is None:
        vmin, vmax = percentiles_vmin_vmax(frame)
    im = ax.imshow(frame, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_overview_grid(ux_samples, uy_samples, times, out_path, cmap="turbo"):
    cols = len(ux_samples)
    fig, axes = plt.subplots(2, cols, figsize=(4*cols, 7))
    axes = np.array(axes)
    for i in range(cols):
        ux = ux_samples[i]
        uy = uy_samples[i]
        vmin_x, vmax_x = percentiles_vmin_vmax(ux)
        vmin_y, vmax_y = percentiles_vmin_vmax(uy)
        axes[0, i].imshow(ux, vmin=vmin_x, vmax=vmax_x, cmap=cmap)
        axes[0, i].set_title(f"t={times[i]} (Ux)", fontsize=11)
        axes[0, i].axis("off")
        axes[1, i].imshow(uy, vmin=vmin_y, vmax=vmax_y, cmap=cmap)
        axes[1, i].set_title(f"t={times[i]} (Uy)", fontsize=11)
        axes[1, i].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize Ux/Uy datasets with optional mask")
    parser.add_argument("--ux", default=DEFAULT_UX_PATH, help="Path to UX_nan_filtered.npy")
    parser.add_argument("--uy", default=DEFAULT_UY_PATH, help="Path to UY_nan_filtered.npy")
    parser.add_argument("--mask", default=None, help="Optional path to mask.npy")
    parser.add_argument("--outdir", default="data_visualization", help="Base output directory")
    parser.add_argument("--samples", type=int, default=12, help="Number of time samples to export")
    parser.add_argument("--cmap", default="turbo", help="Matplotlib colormap to use")
    args = parser.parse_args()

    ux = load_array(args.ux)
    uy = load_array(args.uy)
    if ux.shape != uy.shape:
        raise ValueError(f"Shape mismatch: ux {ux.shape} vs uy {uy.shape}")
    T, nx, ny = ux.shape

    mask2d = load_mask(args.mask, (nx, ny)) if args.mask else None
    if mask2d is not None:
        ux = ux * mask2d
        uy = uy * mask2d

    # stats json
    stats = {
        "ux": {
            "shape": list(ux.shape),
            "min": float(np.min(ux)),
            "max": float(np.max(ux)),
            "mean": float(np.mean(ux)),
            "std": float(np.std(ux)),
        },
        "uy": {
            "shape": list(uy.shape),
            "min": float(np.min(uy)),
            "max": float(np.max(uy)),
            "mean": float(np.mean(uy)),
            "std": float(np.std(uy)),
        },
        "mask": {
            "present": mask2d is not None,
            "shape": list(mask2d.shape) if mask2d is not None else None,
        },
    }
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # choose sample timesteps
    samples = max(1, min(args.samples, T))
    idx_all = np.linspace(0, T - 1, num=samples, dtype=int)

    # export per-frame images
    ux_dir = os.path.join(args.outdir, "ux")
    uy_dir = os.path.join(args.outdir, "uy")
    os.makedirs(ux_dir, exist_ok=True)
    os.makedirs(uy_dir, exist_ok=True)

    ux_samples = []
    uy_samples = []
    for ti in idx_all:
        frame_x = ux[ti]
        frame_y = uy[ti]
        ux_samples.append(frame_x)
        uy_samples.append(frame_y)
        save_frame_image(
            frame_x,
            out_path=os.path.join(ux_dir, f"ux_t{ti:05d}.png"),
            cmap=args.cmap,
            title=f"t={ti} (Ux)"
        )
        save_frame_image(
            frame_y,
            out_path=os.path.join(uy_dir, f"uy_t{ti:05d}.png"),
            cmap=args.cmap,
            title=f"t={ti} (Uy)"
        )

    # overview mosaic
    save_overview_grid(ux_samples, uy_samples, times=list(idx_all), out_path=os.path.join(args.outdir, "overview.png"), cmap=args.cmap)

    print(f"Exported {samples} sampled frames to '{ux_dir}' and '{uy_dir}', overview at '{os.path.join(args.outdir, 'overview.png')}'.")


if __name__ == "__main__":
    main()