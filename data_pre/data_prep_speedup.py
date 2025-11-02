import os
from datetime import datetime

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import multiprocessing as mp

matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "serif"


def interpolate_to_grid_with_mask(grid_x, grid_y, field, NX, NY, grid_xa=None, grid_ya=None, get_mask=False, x_range=None, y_range=None):
    # 保持与原脚本完全一致的处理逻辑（使用全局 X_RANGE/Y_RANGE）
    x_min, x_max = X_RANGE
    y_min, y_max = Y_RANGE
    target_x = np.linspace(x_min, x_max, NX)
    target_y = np.linspace(y_min, y_max, NY)
    grid_X, grid_Y = np.meshgrid(target_x, target_y)

    # 线性插值到结构化网格
    grid_Z = griddata((grid_x, grid_y), field, (grid_X, grid_Y), method="linear")

    if get_mask:
        airfoil_path = Path(np.column_stack((grid_xa, grid_ya)))
        points = np.vstack((grid_X.ravel(), grid_Y.ravel())).T
        inside_airfoil = airfoil_path.contains_points(points).reshape(NY, NX)
        mask = np.where(inside_airfoil, 0, 1)
    else:
        mask = None
    return grid_Z, mask


def fill_nan_with_nearest(data):
    nan_mask = np.isnan(data) | (np.abs(data) > 10)
    distances, indices = distance_transform_edt(nan_mask, return_indices=True)
    filled_data = data[tuple(indices)]
    data[nan_mask] = filled_data[nan_mask]
    return data


# 全局变量（由主进程或初始化函数设置）
GRID_X = None
GRID_Y = None
GRID_XA = None
GRID_YA = None
SCRIPT_DIR = None
NX = None
NY = None
skip_x = None
skip_y = None
X_RANGE = None
Y_RANGE = None


def _init_worker(script_dir, nx, ny, sx, sy, x_range, y_range):
    """每个进程的初始化：加载一次网格数据并设置全局参数。"""
    global GRID_X, GRID_Y, GRID_XA, GRID_YA, SCRIPT_DIR, NX, NY, skip_x, skip_y, X_RANGE, Y_RANGE
    SCRIPT_DIR = script_dir
    NX = nx
    NY = ny
    skip_x = sx
    skip_y = sy
    X_RANGE = x_range
    Y_RANGE = y_range

    # 加载网格数据（保持原路径逻辑）
    f = h5py.File(os.path.join(SCRIPT_DIR, "/data/zhouziyue_benkesheng/downloads/airfoilLES_grid.h5"), "r")
    GRID_X = np.array(f["x"])
    GRID_Y = np.array(f["y"])
    GRID_XA = np.array(f["xa"])
    GRID_YA = np.array(f["ya"])
    f.close()


def _process_file(i):
    """处理单个时间步文件，返回下采样后的 ux/uy 网格以及第一帧的掩码。"""
    t_idx = str(100000 + i)[1:]
    path = os.path.join(SCRIPT_DIR, "/data/zhouziyue_benkesheng/downloads/airfoilLES_midspan", f"airfoilLES_t{t_idx}.h5")
    print(f"reading: {path}")
    f = h5py.File(path, "r")
    ux = np.array(f["ux"])
    uy = np.array(f["uy"])
    f.close()

    # 与原脚本一致的插值及切片逻辑
    field_ux = ux
    if i == 1:
        grid_field_ux, grid_mask = interpolate_to_grid_with_mask(
            GRID_X, GRID_Y, field_ux, NX, NY, GRID_XA, GRID_YA, get_mask=True, x_range=X_RANGE, y_range=Y_RANGE
        )
        grid_mask_down = grid_mask[::skip_x, ::skip_y]
    else:
        grid_field_ux, _ = interpolate_to_grid_with_mask(
            GRID_X, GRID_Y, field_ux, NX, NY, x_range=X_RANGE, y_range=Y_RANGE
        )
        grid_mask_down = None

    temp_grid_field_ux = grid_field_ux[::skip_y, ::skip_x]

    field_uy = uy
    grid_field_uy, _ = interpolate_to_grid_with_mask(
        GRID_X, GRID_Y, field_uy, NX, NY, x_range=X_RANGE, y_range=Y_RANGE
    )
    temp_grid_field_uy = grid_field_uy[::skip_y, ::skip_x]

    return temp_grid_field_ux, temp_grid_field_uy, grid_mask_down


if __name__ == "__main__":
    # 与原脚本一致的参数与输出目录设置
    NX = 512
    NY = 512
    skip_x = skip_y = 2
    X_RANGE = (-0.5, 1.5)
    Y_RANGE = (-1, 1)
    print(NX,NY)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current directory:", script_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, f"results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 读取网格数据用于可视化（保持原行为）
    file = h5py.File(os.path.join(script_dir, "/data/zhouziyue_benkesheng/downloads/airfoilLES_grid.h5"), "r")
    grid_x = np.array(file["x"])
    grid_y = np.array(file["y"])
    grid_xa = np.array(file["xa"])
    grid_ya = np.array(file["ya"])
    file.close()

    plt.scatter(grid_x, grid_y, s=0.01, c="red")
    plt.scatter(grid_xa, grid_ya, s=1, c="blue")
    plt.savefig(os.path.join(output_dir, "grid.png"), dpi=600, bbox_inches="tight")

    # 并行处理所有时间步，保持顺序与首次掩码逻辑一致
    file_index = range(1, 1001)
    ux_ls = []
    uy_ls = []
    grid_mask = None

    # 进程池初始化加载网格数据（每个进程一次），避免在主进程传输大数组
    processes = max(1, (os.cpu_count() or 2) - 1)
    with mp.Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(script_dir, NX, NY, skip_x, skip_y, X_RANGE, Y_RANGE),
    ) as pool:
        for temp_ux, temp_uy, mask_down in tqdm(pool.imap(_process_file, file_index), total=len(file_index)):
            ux_ls.append(temp_ux)
            uy_ls.append(temp_uy)
            if mask_down is not None:
                grid_mask = mask_down

    UX = np.array(ux_ls)
    UY = np.array(uy_ls)
    np.save(os.path.join(output_dir, "UX.npy"), UX)
    np.save(os.path.join(output_dir, "UY.npy"), UY)
    np.save(os.path.join(output_dir, "mask.npy"), grid_mask)

    UX_nan_filtered = fill_nan_with_nearest(UX)
    UY_nan_filtered = fill_nan_with_nearest(UY)
    np.save(os.path.join(output_dir, "UX_nan_filtered.npy"), UX_nan_filtered)
    np.save(os.path.join(output_dir, "UY_nan_filtered.npy"), UY_nan_filtered)