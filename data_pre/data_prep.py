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


matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "serif"


def interpolate_to_grid_with_mask(grid_x, grid_y, field, NX, NY, grid_xa=None, grid_ya=None, get_mask=False, x_range=None, y_range=None):
  
    # x_min = np.min(grid_x) if x_range is None else max(np.min(grid_x), min(x_range))
    # x_max = np.max(grid_x) if x_range is None else min(np.max(grid_x), max(x_range))
    # y_min = np.min(grid_y) if y_range is None else max(np.min(grid_y), min(y_range))
    # y_max = np.max(grid_y) if y_range is None else min(np.max(grid_y), max(y_range))
    x_min,x_max = X_RANGE
    y_min,y_max = Y_RANGE
    target_x = np.linspace(x_min, x_max, NX)
    target_y = np.linspace(y_min, y_max, NY)
    grid_X, grid_Y = np.meshgrid(target_x, target_y)
    #核心：非结构化网格映射到结构化网格内
    """
    如果用 “最近邻插值（method='nearest'）”：
    只看P离A和B哪个更近，取更近的那个点的值。比如P离A更近，P的值就取 10。
    如果用 “线性插值（method='linear'）”：
    会根据P到A、B的距离分配权重（离得越近权重越大），计算加权平均。比如P到A的距离是 1，到B的距离是 2，那么P的值≈(10×2 + 20×1)/(1+2)≈13.3。
    如果用 “三次插值（method='cubic'）”：
    会考虑更多周围点（不止A和B）的分布，用更复杂的曲线拟合计算P的值，结果更平滑，但依然是综合周围点的信息。

    结构化网格点不在非结构化点的 “覆盖范围” 内（比如远离所有测量点的区域）:
    默认返回NaN（Not a Number），表示 “该网格点的数值无法通过已知的非结构化点插值得到”。
    """
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

#10.28之前的参数
# NX = 512 #这里设计的核心是保证在映射的时候，原来的数据不会被压缩
# NY = 512 #这里设计的核心是保证在映射的时候，原来的数据不会被压缩
# skip_x = skip_y = 2#间隔采样，不然矩阵太大了
# # 可选：物理坐标裁剪范围（设为 None 表示不裁剪）
# X_RANGE = (-0.75,1.75)
# Y_RANGE = (-2,2)

#10.28之后的参数
NX = 512 #这里设计的核心是保证在映射的时候，原来的数据不会被压缩
NY = 512 #这里设计的核心是保证在映射的时候，原来的数据不会被压缩
skip_x = skip_y = 2#间隔采样，不然矩阵太大了
# 可选：物理坐标裁剪范围（设为 None 表示不裁剪）
X_RANGE = (-0.5,1.5)
Y_RANGE = (-1,1)
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Current directory:", script_dir)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(script_dir, f"results_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

""" 读取网格数据 并可视化网格"""
file = h5py.File(os.path.join(script_dir, "/data/zhouziyue_benkesheng/downloads/airfoilLES_grid.h5"), "r")
grid_x = np.array(file["x"])
grid_y = np.array(file["y"])
grid_xa = np.array(file["xa"])
grid_ya = np.array(file["ya"])
plt.scatter(grid_x, grid_y, s=0.01, c="red")
plt.scatter(grid_xa, grid_ya, s=1, c="blue")
plt.savefig(os.path.join(output_dir, "grid.png"), dpi=600, bbox_inches="tight")

ux_ls = []
uy_ls = []

# for i in tqdm(range(1, 3901, 5)):
file_index=range(1,501)
for i in tqdm(file_index):#这里设计到采样的问题，先不间隔采样了
    t_idx = str(100000 + i)[1:]
    file = h5py.File(
        os.path.join(script_dir, "/data/zhouziyue_benkesheng/downloads/airfoilLES_midspan", f"airfoilLES_t{t_idx}.h5"),
        "r",
    )
    print(f"reading: {file}")

    ux = np.array(file["ux"])
    uy = np.array(file["uy"])

    field_ux = ux
    if i == 1:
        grid_field_ux, grid_mask = interpolate_to_grid_with_mask(
            grid_x, grid_y, field_ux, NX, NY, grid_xa, grid_ya, get_mask=True, x_range=X_RANGE, y_range=Y_RANGE
        )
        grid_mask = grid_mask[::skip_x, ::skip_y] 
        
    else:
        grid_field_ux, _ = interpolate_to_grid_with_mask(grid_x, grid_y, field_ux, NX, NY, x_range=X_RANGE, y_range=Y_RANGE)
    temp_grid_field_ux = grid_field_ux[::skip_y, ::skip_x]
    ux_ls.append(temp_grid_field_ux)

    field_uy = uy
    grid_field_uy, _ = interpolate_to_grid_with_mask(grid_x, grid_y, field_uy, NX, NY, x_range=X_RANGE, y_range=Y_RANGE)
    temp_grid_field_uy = grid_field_uy[::skip_y, ::skip_x]
    uy_ls.append(temp_grid_field_uy)

    file.close()

UX = np.array(ux_ls)
UY = np.array(uy_ls)
np.save(os.path.join(output_dir, "UX.npy"), UX)
np.save(os.path.join(output_dir, "UY.npy"), UY)
np.save(os.path.join(output_dir, "mask.npy"), grid_mask)
UX_nan_filtered = fill_nan_with_nearest(UX)
UY_nan_filtered = fill_nan_with_nearest(UY)

np.save(os.path.join(output_dir, "UX_nan_filtered.npy"), UX_nan_filtered)
np.save(os.path.join(output_dir, "UY_nan_filtered.npy"), UY_nan_filtered)
# 合并为最后维度=2: [..., 0]=ux, [..., 1]=uy

# U_XY_filtered = np.stack((UX_nan_filtered, UY_nan_filtered), axis=-1)
# np.save(os.path.join(script_dir, "U_XY_filtered.npy"), U_XY_filtered)