import os
from typing import Optional, Tuple

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无GUI环境下保存图片
import matplotlib.pyplot as plt

GRID_FILE = r"c:\Users\Harry\Desktop\project\ana_h5\airfoilLES_grid.h5"
SAVE_PATH = r"c:\Users\Harry\Desktop\project\ana_h5\grid_airfoil.png"


def find_dataset(file: h5py.File, names: Tuple[str, ...]) -> Optional[h5py.Dataset]:
    """在H5文件中按候选名字查找数据集（大小写不敏感，支持路径包含与末段匹配）。"""
    all_dsets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_dsets.append((name, obj))

    file.visititems(visitor)

    lowers = tuple(n.lower() for n in names)
    # 先尝试末段精确匹配
    for path, ds in all_dsets:
        base = path.split("/")[-1].lower()
        if base in lowers:
            return ds
    # 再尝试包含匹配
    for path, ds in all_dsets:
        base = path.split("/")[-1].lower()
        if any(base == n or base.endswith(n) or base.startswith(n) for n in lowers):
            return ds
    # 最后尝试路径包含
    for path, ds in all_dsets:
        p = path.lower()
        if any(n in p for n in lowers):
            return ds
    return None


def plot_points_with_airfoil(x_ds: h5py.Dataset, y_ds: h5py.Dataset,
                             xa_ds: Optional[h5py.Dataset], ya_ds: Optional[h5py.Dataset],
                             save_path: str) -> None:
    """按(x[i], y[i])散点绘制网格节点，并叠加机翼外形(xa, ya)。"""
    # 读出并处理形状
    x = np.array(x_ds[()]).astype(np.float64, copy=False)
    y = np.array(y_ds[()]).astype(np.float64, copy=False)

    # 支持二维/更高维数据，统一拉平成一维
    if x.ndim > 1:
        x = x.reshape(-1)
    if y.ndim > 1:
        y = y.reshape(-1)

    if x.shape != y.shape:
        raise ValueError(f"x与y长度不一致: {x.shape} vs {y.shape}")

    npts = x.size
    # 如点数极大，做可视化抽样（当前数据约22万点，直接绘制即可）
    max_draw = 1_000_000
    if npts > max_draw:
        step = int(np.ceil(npts / max_draw))
        x = x[::step]
        y = y[::step]
        npts = x.size

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    ax.scatter(x, y, s=0.4, c="#1f77b4", alpha=0.7, edgecolors="none", label=f"网格点 ({npts})")

    # 叠加机翼外形（如果提供）
    if xa_ds is not None and ya_ds is not None:
        xa = np.array(xa_ds[()]).astype(np.float64, copy=False)
        ya = np.array(ya_ds[()]).astype(np.float64, copy=False)
        if xa.ndim > 1:
            xa = xa.reshape(-1)
        if ya.ndim > 1:
            ya = ya.reshape(-1)
        ax.plot(xa, ya, color="#d62728", linewidth=1.2, label="Airfoil")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Airfoil Surrounding Grid Points")
    ax.legend(loc="best")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    grid_path = GRID_FILE
    save_path = SAVE_PATH
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"未找到网格文件: {grid_path}")

    with h5py.File(grid_path, "r") as f:
        # 查找x,y网格坐标与机翼外形xa,ya
        x_ds = find_dataset(f, ("x", "grid_x", "X"))
        y_ds = find_dataset(f, ("y", "grid_y", "Y"))
        xa_ds = find_dataset(f, ("xa", "airfoil_x", "XA"))
        ya_ds = find_dataset(f, ("ya", "airfoil_y", "YA"))

        if x_ds is None or y_ds is None:
            raise KeyError("未找到数据集x或y，请确认数据集名称或路径包含x/y")

        plot_points_with_airfoil(x_ds, y_ds, xa_ds, ya_ds, save_path)
        print(f"已保存散点网格图（含机翼外形）: {save_path}")


if __name__ == "__main__":
    main()