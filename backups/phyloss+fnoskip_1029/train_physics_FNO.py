import json
import logging
import math
import os
import random
import sys
import time
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paddle
import scipy.stats as stats
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from model.unet_withFNO import Unet2D_with_FNO


def setup_seed(seed):
    paddle.seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    paddle.set_flags({"FLAGS_cudnn_deterministic": True})
    paddle.set_flags({"FLAGS_benchmark": False})

#配置logger的
def init_all(seed, name, dtype):
    setup_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    paddle.set_default_dtype(d=dtype)

    # 设备选择：若存在可用 GPU，则切换到 GPU，否则使用 CPU
    def select_best_gpu():
        # 优先使用 NVML 查询内存与利用率，失败则回退到 nvidia-smi
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            best_idx = 0
            best_score = float("inf")
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                # 评分：内存占用为主，利用率为辅
                score = (mem.used / max(mem.total, 1)) + (util / 100.0) * 0.5
                if score < best_score:
                    best_score = score
                    best_idx = i
            pynvml.nvmlShutdown()
            return best_idx
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
                errors="ignore",
            )
            best_idx = 0
            best_score = float("inf")
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    i = int(parts[0])
                    util = float(parts[1])
                    mem_used = float(parts[2])
                    mem_total = float(parts[3])
                    score = (mem_used / max(mem_total, 1)) + (util / 100.0) * 0.5
                    if score < best_score:
                        best_score = score
                        best_idx = i
            return best_idx
        except Exception:
            pass
        return 0

    try:
        gpu_available = bool(paddle.is_compiled_with_cuda())
        try:
            gpu_count = paddle.device.cuda.device_count()
        except Exception:
            gpu_count = 0
        if gpu_available and gpu_count > 0:
            chosen_idx = select_best_gpu()
            device = f"gpu:{chosen_idx}"
        else:
            device = "cpu"
        paddle.device.set_device(device)
    except Exception as e:
        device = f"cpu (fallback: {e})"

    if not os.path.exists(name):
        os.makedirs(name)
    log_level = logging.INFO
    log_name = os.path.join(name, time.strftime("%Y-%m-%d-%H-%M-%S") + ".log")
    logger = logging.getLogger("")
    logger.setLevel(log_level)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_name, encoding="utf8")
    file_handler.setLevel(level=log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Project name: {name}")
    logger.info(f"Random seed value: {seed}, data type : {dtype}")
    logger.info(f"Device: {device}")
    logger.info(f"CUDA available: {gpu_available}, GPU count: {gpu_count}\n")
    return logger


def make_plot(TRUE, PRED, epoch, images_dir="images", train_losses=None, val_losses=None, val_rollout_losses=None, val_1step_losses=None, mask2d=None, zero_center=True, unify_scale=False):
    """
    可视化验证结果（旧版增强版）：
    - 每列对应一个被选时间点，按该列帧的值域自适应设定色阶，避免整片绿色。
    - 计算并在标题显示相对 L2 误差（返回 float）。
    - 进行频谱对比：2D FFT -> 能量谱 -> 分箱 -> 对数坐标绘制，可叠加 `k^-5/3` 参考线。
    - 支持传入二维掩膜 `mask2d`，可选零中心/共用色阶显示方式。
    - 每个 epoch 保存图像到指定的 images 目录。
    """
    sample_id = 0
    T = TRUE.shape[1]
    skip_t = max(1, T // 8)
    idx_all = np.arange(T)[::skip_t]
    N = min(idx_all.shape[0], 3)
    idx = idx_all[:N]

    true = np.nan_to_num(TRUE[sample_id, idx, 0], nan=0.0, posinf=0.0, neginf=0.0)
    pred1 = np.nan_to_num(PRED[sample_id, idx, 0], nan=0.0, posinf=0.0, neginf=0.0)
    true_uy = np.nan_to_num(TRUE[sample_id, idx, 1], nan=0.0, posinf=0.0, neginf=0.0)
    pred_uy = np.nan_to_num(PRED[sample_id, idx, 1], nan=0.0, posinf=0.0, neginf=0.0)
    time_ls = idx
    CMAP = "turbo"

    def rel_l2(Ta, Pa):
        try:
            dtype_str = paddle.get_default_dtype()
            if not isinstance(dtype_str, str):
                dtype_str = 'float32'
        except Exception:
            dtype_str = 'float32'
        Tt = paddle.to_tensor(data=Ta, dtype=dtype_str)
        Pt = paddle.to_tensor(data=Pa, dtype=dtype_str)
        numer = paddle.linalg.norm(x=Tt - Pt, p=2)
        denom = paddle.linalg.norm(x=Tt, p=2)
        val = numer / paddle.clip(denom, min=1e-8)
        return float(val.item())

    def frame_vmin_vmax(x):
        vals = np.asarray(x).flatten()
        vals = vals[np.isfinite(vals)]
        vals = vals[np.abs(vals) > 1e-8]
        if vals.size < 50:
            vals = np.asarray(x).flatten()
            vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vmin = float(np.nanmin(x))
            vmax = float(np.nanmax(x))
            return vmin, vmax
        vmin = np.percentile(vals, 1)
        vmax = np.percentile(vals, 99)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            m = float(vals.mean())
            s = float(vals.std())
            if s > 0:
                vmin = m - 3.0 * s
                vmax = m + 3.0 * s
            else:
                vmin = float(vals.min())
                vmax = float(vals.max())
        if vmin == vmax:
            eps = 1e-12 if vmax == 0 else abs(vmax) * 1e-6
            vmin -= eps
            vmax += eps
        return float(vmin), float(vmax)

    fig, axes = plt.subplots(6, N, figsize=(30, 14))
    axes = np.array(axes)
    cbar_ax = fig.add_axes([0.92, 0.26, 0.02, 0.62])

    m2d = None
    if mask2d is not None:
        m2d = np.asarray(mask2d)
        if m2d.ndim > 2:
            m2d = np.squeeze(m2d)
        m2d = m2d.astype(np.float32)

    for i in range(N):
        t_frame = true[i]
        p_frame = pred1[i]
        if m2d is not None:
            t_frame = t_frame * m2d
            p_frame = p_frame * m2d
        vmin_t, vmax_t = frame_vmin_vmax(t_frame)
        vmin_p, vmax_p = frame_vmin_vmax(p_frame)
        if zero_center:
            max_abs_t = max(abs(vmin_t), abs(vmax_t))
            max_abs_p = max(abs(vmin_p), abs(vmax_p))
            max_abs_t = max(max_abs_t, 1e-12)
            max_abs_p = max(max_abs_p, 1e-12)
            vmin_t, vmax_t = -max_abs_t, max_abs_t
            vmin_p, vmax_p = -max_abs_p, max_abs_p
        if unify_scale:
            max_abs_pair = max(abs(vmin_t), abs(vmax_t), abs(vmin_p), abs(vmax_p))
            max_abs_pair = max(max_abs_pair, 1e-12)
            vmin_t = vmin_p = -max_abs_pair
            vmax_t = vmax_p = max_abs_pair
        im = axes[0, i].imshow(t_frame, vmin=vmin_t, vmax=vmax_t, cmap=CMAP)
        axes[0, i].set_title(f"Time: {int(time_ls[i])}s (Ux)", fontsize=14)
        axes[0, i].axis("off")

        im = axes[1, i].imshow(p_frame, vmin=vmin_p, vmax=vmax_p, cmap=CMAP)
        mse_val1 = rel_l2(t_frame, p_frame)
        axes[1, i].set_title(f"rel L2 (Ux): {mse_val1:.2e}", fontsize=12)
        axes[1, i].axis("off")

        image = true[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2).flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        kbins = np.arange(0.5, min(nx, ny) // 2 + 1, 1.0)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins_true, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins_true *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_true = np.maximum(Abins_true, 1e-12)
        axes[2, i].loglog(kvals, Abins_true, 'b-', label="True Ux", linewidth=2)

        image = pred1[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2).flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        Abins_pred, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins_pred *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_pred = np.maximum(Abins_pred, 1e-12)
        axes[2, i].loglog(kvals, Abins_pred, 'r--', label="Pred Ux", linewidth=2)

        k_ref = np.linspace(1, np.max(kbins), 100)
        energy_ref = k_ref ** (-5 / 3)
        energy_ref *= max(max(Abins_true), max(Abins_pred)) / max(energy_ref)
        axes[2, i].loglog(k_ref, energy_ref, "k:", label="k^-5/3", alpha=0.7)
        axes[2, i].legend(fontsize=9)
        axes[2, i].set_xlabel("$k$")
        axes[2, i].set_title("Ux Energy Spectrum", fontsize=12)

        ty_frame = true_uy[i]
        py_frame = pred_uy[i]
        if m2d is not None:
            ty_frame = ty_frame * m2d
            py_frame = py_frame * m2d
        vmin_ty, vmax_ty = frame_vmin_vmax(ty_frame)
        vmin_py, vmax_py = frame_vmin_vmax(py_frame)
        if zero_center:
            max_abs_ty = max(abs(vmin_ty), abs(vmax_ty))
            max_abs_py = max(abs(vmin_py), abs(vmax_py))
            max_abs_ty = max(max_abs_ty, 1e-12)
            max_abs_py = max(max_abs_py, 1e-12)
            vmin_ty, vmax_ty = -max_abs_ty, max_abs_ty
            vmin_py, vmax_py = -max_abs_py, max_abs_py
        if unify_scale:
            max_abs_pair_y = max(abs(vmin_ty), abs(vmax_ty), abs(vmin_py), abs(vmax_py))
            max_abs_pair_y = max(max_abs_pair_y, 1e-12)
            vmin_ty = vmin_py = -max_abs_pair_y
            vmax_ty = vmax_py = max_abs_pair_y
        im2 = axes[3, i].imshow(ty_frame, vmin=vmin_ty, vmax=vmax_ty, cmap=CMAP)
        axes[3, i].set_title(f"Time: {int(time_ls[i])}s (Uy)", fontsize=14)
        axes[3, i].axis("off")

        im2 = axes[4, i].imshow(py_frame, vmin=vmin_py, vmax=vmax_py, cmap=CMAP)
        mse_val_u = rel_l2(ty_frame, py_frame)
        axes[4, i].set_title(f"rel L2 (Uy): {mse_val_u:.2e}", fontsize=12)
        axes[4, i].axis("off")

        image = true_uy[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2).flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        Abins_true, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins_true *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_true = np.maximum(Abins_true, 1e-12)

        image = pred_uy[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2).flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        Abins_pred, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins_pred *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_pred = np.maximum(Abins_pred, 1e-12)

        axes[5, i].loglog(kvals, Abins_true, label="Simulated Uy", color="tab:blue")
        axes[5, i].loglog(kvals, Abins_pred, label="MATCHO Uy", color="tab:orange")
        k_ref = np.linspace(1, np.max(kbins), 100)
        energy_ref = k_ref ** (-5 / 3)
        energy_ref *= max(max(Abins_true), max(Abins_pred)) / max(energy_ref)
        axes[5, i].loglog(k_ref, energy_ref, "k--", label="k^-5/3")
        axes[5, i].legend(fontsize=10)
        axes[5, i].set_title(f"Uy Energy Spectrum (t={int(time_ls[i])})")
        axes[5, i].set_xlabel("$k$")
        axes[5, i].set_ylabel("Energy")

    fig.subplots_adjust(left=0.06, right=0.9, top=0.96, bottom=0.22, hspace=0.8, wspace=0.25)
    if (val_rollout_losses is None) and (val_losses is not None):
        val_rollout_losses = val_losses
    if train_losses is not None and val_rollout_losses is not None:
        try:
            loss_ax = fig.add_axes([0.08, 0.04, 0.8, 0.14])
            x = np.arange(1, len(train_losses) + 1)
            val_x = np.arange(1, len(val_rollout_losses) + 1)
            train_arr = np.asarray(train_losses, dtype=float)
            val_arr = np.asarray(val_rollout_losses, dtype=float)
            val1_arr = np.asarray(val_1step_losses, dtype=float) if val_1step_losses is not None else None
            train_arr = np.where(np.isfinite(train_arr), train_arr, np.nan)
            val_arr = np.where(np.isfinite(val_arr), val_arr, np.nan)
            if val1_arr is not None:
                val1_arr = np.where(np.isfinite(val1_arr), val1_arr, np.nan)

            has_train_pos = np.any((train_arr > 0) & np.isfinite(train_arr))
            has_val_pos = np.any((val_arr > 0) & np.isfinite(val_arr))
            has_val1_pos = np.any((val1_arr > 0) & np.isfinite(val1_arr)) if val1_arr is not None else False
            use_log = bool(has_train_pos or has_val_pos or has_val1_pos)

            if use_log:
                loss_ax.set_yscale("log")
                train_plot = np.where(train_arr > 0, train_arr, np.nan)
                val_plot = np.where(val_arr > 0, val_arr, np.nan)
                loss_ax.plot(x, train_plot, label="Train loss", color="tab:blue", linewidth=2)
                loss_ax.plot(val_x, val_plot, label="Val rollout loss", color="tab:orange", linestyle="--", marker="o", markersize=3, alpha=0.9, linewidth=2, zorder=3)
                if val1_arr is not None:
                    val1_x = np.arange(1, len(val1_arr) + 1)
                    val1_plot = np.where(val1_arr > 0, val1_arr, np.nan)
                    loss_ax.plot(val1_x, val1_plot, label="Val 1-step loss", color="tab:green", linestyle=":", marker="s", markersize=3, alpha=0.9, linewidth=2, zorder=2)
                pos_vals = np.concatenate([
                    train_arr[(train_arr > 0) & np.isfinite(train_arr)],
                    val_arr[(val_arr > 0) & np.isfinite(val_arr)],
                    val1_arr[(val1_arr > 0) & np.isfinite(val1_arr)] if val1_arr is not None else np.array([], dtype=float),
                ])
                if pos_vals.size > 0:
                    y_min = float(np.nanmin(pos_vals))
                    y_max = float(np.nanmax(pos_vals))
                    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > 0:
                        loss_ax.set_ylim(max(y_min, 1e-12), y_max * 1.2)
                loss_ax.set_ylabel("Loss (log)")
            else:
                loss_ax.set_yscale("linear")
                loss_ax.plot(x, train_arr, label="Train loss", color="tab:blue", linewidth=2)
                loss_ax.plot(val_x, val_arr, label="Val rollout loss", color="tab:orange", linestyle="--", marker="o", markersize=3, alpha=0.9, linewidth=2, zorder=3)
                if val1_arr is not None:
                    val1_x = np.arange(1, len(val1_arr) + 1)
                    loss_ax.plot(val1_x, val1_arr, label="Val 1-step loss", color="tab:green", linestyle=":", marker="s", markersize=3, alpha=0.9, linewidth=2, zorder=2)
                loss_ax.set_ylabel("Loss")

            loss_ax.set_xlabel("Epoch")
            loss_ax.set_title(f"Loss curves through epoch {epoch + 1}")
            loss_ax.grid(True, which="both", ls=":", alpha=0.3)
            loss_ax.legend(loc="upper right")
        except Exception:
            pass
 
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(os.path.join(images_dir, f"{epoch + 1}.png"))
    plt.close(fig)


class CustomLoss(paddle.nn.Layer):
    def __init__(self, Par, mask=None):
        super(CustomLoss, self).__init__()
        self.Par = Par
        # 添加物理约束的权重参数
        self.mse_weight = Par.get('mse_weight', 1.0)
        self.div_weight = Par.get('div_weight', 0.1)  # 连续性方程权重
        self.momentum_weight = Par.get('momentum_weight', 0.05)  # 动量方程权重
        # 物理项 warmup 因子（训练早期逐步引入物理约束，提升稳定性）
        self._warmup = 1.0
        # 流体参数
        self.Ma = 0.3  # 马赫数
        self.Re = 23000  # 雷诺数
        self.nu = 1.0 / self.Re  # 运动粘度
        # 机翼掩码，用于排除机翼区域
        self.mask = mask

    def set_warmup_factor(self, factor: float):
        try:
            factor = float(factor)
        except Exception:
            factor = 1.0
        self._warmup = max(0.0, min(1.0, factor))
    # 将通道维度规范为最后一维，便于统一差分计算
    def _to_channel_last(self, tensor):
        shape = list(tensor.shape)
        if len(shape) == 0:
            return tensor
        if shape[-1] == 2:
            return tensor
        # [B, C, H, W]
        if len(shape) == 4 and shape[1] == 2:
            return paddle.transpose(tensor, perm=[0, 2, 3, 1])
        # [B, T, C, H, W]
        if len(shape) == 5 and shape[2] == 2:
            return paddle.transpose(tensor, perm=[0, 1, 3, 4, 2])
        # 通用：把第一个维度为2的轴移到最后
        if 2 in shape:
            c_axis = shape.index(2)
            perm = [i for i in range(len(shape)) if i != c_axis] + [c_axis]
            return paddle.transpose(tensor, perm=perm)
        return tensor

    # x方向一阶差分（最后一维）
    def _ddx(self, u):
        left = u[..., :, 1:2] - u[..., :, 0:1]
        center = (u[..., :, 2:] - u[..., :, :-2]) * 0.5
        right = u[..., :, -1:] - u[..., :, -2:-1]
        return paddle.concat([left, center, right], axis=-1)

    # y方向一阶差分（倒数第二维）
    def _ddy(self, u):
        top = u[..., 1:2, :] - u[..., 0:1, :]
        center = (u[..., 2:, :] - u[..., :-2, :]) * 0.5
        bottom = u[..., -1:, :] - u[..., -2:-1, :]
        return paddle.concat([top, center, bottom], axis=-2)

    # x方向二阶差分
    def _d2dx2(self, u):
        left = u[..., :, 1:2] - 2.0 * u[..., :, 0:1] + u[..., :, 0:1]
        center = u[..., :, 2:] - 2.0 * u[..., :, 1:-1] + u[..., :, :-2]
        right = u[..., :, -1:] - 2.0 * u[..., :, -1:] + u[..., :, -2:-1]
        return paddle.concat([left, center, right], axis=-1)

    # y方向二阶差分
    def _d2dy2(self, u):
        top = u[..., 1:2, :] - 2.0 * u[..., 0:1, :] + u[..., 0:1, :]
        center = u[..., 2:, :] - 2.0 * u[..., 1:-1, :] + u[..., :-2, :]
        bottom = u[..., -1:, :] - 2.0 * u[..., -1:, :] + u[..., -2:-1, :]
        return paddle.concat([top, center, bottom], axis=-2)

    def forward(self, y_pred, y_true):
        # 数据反归一化
        y_true = (y_true - self.Par["out_shift"]) / (self.Par["out_scale"])
        y_pred = (y_pred - self.Par["out_shift"]) / (self.Par["out_scale"])

        # 数值安全处理：替换 NaN/Inf，避免非有限损失
        y_true = paddle.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = paddle.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        # 统一通道为最后一维，支持 [B,T,C,H,W]/[B,C,H,W]/[B,H,W,C]
        y_true = self._to_channel_last(y_true)
        y_pred = self._to_channel_last(y_pred)

        # 1. MSE损失 - 只计算流体区域（使用掩码）
        if self.mask is not None:
            mask_tensor = paddle.to_tensor(self.mask, dtype=y_pred.dtype)
            # 将掩码沿前导维度扩展到与张量空间维匹配（忽略通道维）
            while len(mask_tensor.shape) < len(y_pred.shape) - 1:
                mask_tensor = paddle.unsqueeze(mask_tensor, axis=0)
            # 在最后一维为通道添加单维，确保广播到 C 通道
            mask_tensor = paddle.unsqueeze(mask_tensor, axis=-1)
            # 计算掩码区域的MSE损失（按通道加权平均）
            squared_diff = (y_true - y_pred) ** 2
            masked_squared_diff = squared_diff * mask_tensor
            denom = paddle.sum(mask_tensor) * float(y_pred.shape[-1])
            denom = paddle.clip(denom, min=1e-12)
            mse_loss = paddle.sum(masked_squared_diff) / denom
        else:
            mse_loss = paddle.mean((y_true - y_pred) ** 2)

        physics_loss = paddle.to_tensor(0.0, dtype=y_pred.dtype)

        # 计算速度场的散度作为连续性方程约束
        if self.div_weight > 0:
            ux = y_pred[..., 0]
            uy = y_pred[..., 1]
            # 对速度进行合理范围钳制，缓解早期爆炸导致的非有限梯度
            try:
                ux = paddle.clip(ux, min=-10.0, max=10.0)
                uy = paddle.clip(uy, min=-10.0, max=10.0)
            except Exception:
                pass
            divergence = self._ddx(ux) + self._ddy(uy)
            if self.mask is not None:
                mask_tensor = paddle.to_tensor(self.mask, dtype=divergence.dtype)
                while len(mask_tensor.shape) < len(divergence.shape):
                    mask_tensor = paddle.unsqueeze(mask_tensor, axis=0)
                div_loss = paddle.sum((divergence ** 2) * mask_tensor) / paddle.sum(mask_tensor)
            else:
                div_loss = paddle.mean(divergence ** 2)
            physics_loss = physics_loss + (self.div_weight * self._warmup) * div_loss

        # 简化的动量方程约束（忽略压力项）
        if self.momentum_weight > 0:
            ux = y_pred[..., 0]
            uy = y_pred[..., 1]
            try:
                ux = paddle.clip(ux, min=-10.0, max=10.0)
                uy = paddle.clip(uy, min=-10.0, max=10.0)
            except Exception:
                pass
            dudx = self._ddx(ux)
            dudy = self._ddy(ux)
            dvdx = self._ddx(uy)
            dvdy = self._ddy(uy)
            d2udx2 = self._d2dx2(ux)
            d2udy2 = self._d2dy2(ux)
            d2vdx2 = self._d2dx2(uy)
            d2vdy2 = self._d2dy2(uy)
            convective_x = ux * dudx + uy * dudy
            convective_y = ux * dvdx + uy * dvdy
            viscous_x = self.nu * (d2udx2 + d2udy2)
            viscous_y = self.nu * (d2vdx2 + d2vdy2)
            momentum_residual_x = convective_x - viscous_x
            momentum_residual_y = convective_y - viscous_y
            momentum_residual = momentum_residual_x ** 2 + momentum_residual_y ** 2
            if self.mask is not None:
                mask_tensor = paddle.to_tensor(self.mask, dtype=momentum_residual.dtype)
                while len(mask_tensor.shape) < len(momentum_residual.shape):
                    mask_tensor = paddle.unsqueeze(mask_tensor, axis=0)
                momentum_loss = paddle.sum(momentum_residual * mask_tensor) / paddle.sum(mask_tensor)
            else:
                momentum_loss = paddle.mean(momentum_residual)
            physics_loss = physics_loss + (self.momentum_weight * self._warmup) * momentum_loss

        total_loss = self.mse_weight * mse_loss + physics_loss

        if hasattr(self, 'loss_components'):
            try:
                self.loss_components = {
                    'mse_loss': mse_loss.numpy(),
                    'physics_loss': physics_loss.numpy(),
                }
            except Exception:
                pass
        return total_loss


class YourDataset_train(paddle.io.Dataset):
    def __init__(self, x, t, y, transform=None):
        self.x = x
        self.t = t
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        t_sample = self.t[idx]
        y_sample = self.y[idx]
        if self.transform:
            x_sample, t_sample, y_sample = self.transform(x_sample, t_sample, y_sample)
        return x_sample, t_sample, y_sample


class YourDataset(paddle.io.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]
        if self.transform:
            x_sample, y_sample = self.transform(x_sample, y_sample)
        return x_sample, y_sample


def preprocess_train(traj, Par):
    """
    构造训练样本的索引：
    - `x_idx`：时间滑动窗口（长度 `lb`），作为模型输入的时间索引。
    - `t_idx`：长度 `lf` 的时间条件索引，用于传入模型的条件向量索引。
    - `y_idx`：目标窗口起止索引（长度 `lf`），对应监督的真值时间点。
    返回三者的 `int64` Tensor 索引，供 `YourDataset_train` 使用。
    """
    nt = traj.shape[1]
    temp = nt - Par["lb"] - Par["lf"] + 1
    x_idx = np.arange(temp).reshape(-1, 1)
    x_idx = np.tile(x_idx, (1, Par["lf"]))
    x_idx = x_idx.reshape(-1, 1)
    x_idx_ls = []
    for i in range(Par["lb"]):
        x_idx_ls.append(x_idx + i)
    x_idx = np.concatenate(x_idx_ls, axis=1)
    t_idx = np.arange(Par["lf"]).reshape(1, -1)
    t_idx = np.tile(t_idx, (temp, 1)).reshape(-1)
    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par["lb"] :], window_shape=Par["lf"]).reshape(-1)
    return (
        paddle.to_tensor(data=x_idx, dtype="int64"),
        paddle.to_tensor(data=t_idx, dtype="int64"),
        paddle.to_tensor(data=y_idx, dtype="int64"),
    )


def preprocess(traj, Par):
    """
    构造验证/测试样本的索引（自适应窗口）：
    - `x_idx`：时间滑动窗口（长度 `lb`），作为模型输入的时间索引。
    - `t_idx`：长度 `lf` 的时间条件索引。
    - `y_idx`：长序列目标窗口（长度自适应为 `min(LF, nt - lb)`），用于 `rollout` 的连续评估。
    当验证/测试序列较短，自动缩小窗口长度，避免窗口超过输入长度。
    """
    nt = traj.shape[1]
    # 至少需要 nt - lb >= 1 才能形成监督目标
    if nt - Par["lb"] < 1:
        raise ValueError(f"序列过短：nt={nt}, lb={Par['lb']}，至少需要 nt - lb >= 1")
    effective_LF = min(Par["LF"], nt - Par["lb"])  # 自适应窗口长度
    temp = nt - Par["lb"] - effective_LF + 1
    x_idx = np.arange(temp).reshape(-1, 1)
    x_idx_ls = []
    for i in range(Par["lb"]):
        x_idx_ls.append(x_idx + i)
    x_idx = np.concatenate(x_idx_ls, axis=1)
    t_idx = np.arange(Par["lf"]).reshape(-1)
    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par["lb"] :], window_shape=effective_LF)
    return (
        paddle.to_tensor(data=x_idx, dtype="int64"),
        paddle.to_tensor(data=t_idx, dtype="int64"),
        paddle.to_tensor(data=y_idx, dtype="int64"),
    )


def combined_scheduler(optimizer, total_epochs, warmup_epochs, last_epoch=-1):
    """
    组合学习率调度（warmup + 余弦退火）：
    - 前 `warmup_epochs`（按批计算）线性升温。
    - 之后按余弦退火逐步降低到 0。
    训练循环内每个批次调用 `scheduler.step()` 完成学习率更新。
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    tmp_lr = paddle.optimizer.lr.LambdaDecay(
        lr_lambda=lr_lambda, last_epoch=last_epoch, learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(tmp_lr)
    return tmp_lr


def rollout(model, x, t, NT, Par, batch_size):
    """
    长序列滚动预测：
    - 以输入窗口 `lb` 帧起始，模型每次生成 `lf` 帧并追加到轨迹。
    - 每次滑窗更新下一次输入，直至累计到 `NT = lb + LF` 帧。
    - 返回预测序列，形状为 `[batch, LF, nf, nx, ny]`，用于验证/测试对比。
    """
    y_pred_ls = []
    bs = batch_size
    end = bs

    while True:
        start = end - bs
        out_ls = []

        if start >= x.shape[0]:
            break
        temp_x1 = x[start:end]
        out_ls = [temp_x1]
        traj = paddle.concat(x=out_ls, axis=1)

        while traj.shape[1] < NT:
            with paddle.no_grad ():
                temp_x = paddle.repeat_interleave(x=temp_x1, repeats=Par["lf"], axis=0)
                temp_t = t.tile(repeat_times=traj.shape[0])
                out = model(temp_x, temp_t).reshape([-1, Par["lf"], Par["nf"], Par["nx"], Par["ny"]])
                out_ls.append(out)
                traj = paddle.concat(x=out_ls, axis=1)
                temp_x1 = traj[:, -Par["lb"] :]
        pred = paddle.concat(x=out_ls, axis=1)[:, Par["lb"] : NT]
        y_pred_ls.append(pred)
        end = end + bs
        if end - bs > x.shape[0] + 1:
            break

    if len(y_pred_ls) > 0:
        y_pred = paddle.concat(y_pred_ls, axis=0)
    else:
        y_pred = paddle.zeros([0, NT - Par["lb"], Par["nf"], Par["nx"], Par["ny"]])
    return y_pred


if __name__ == "__main__":

    #创建好日志对象
    seed_value = 23
    data_type = "float32"
    
    # 创建带时间戳的主文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    main_output_dir = f"train_results_{timestamp}"
    os.makedirs(main_output_dir, exist_ok=True)
    # 初始化日志目录并创建 logger（必须在首次使用前）
    save_dir = os.path.join(main_output_dir, f"seed_{seed_value}")
    logger = init_all(seed_value, name=save_dir, dtype=data_type)

    ux_data_dir = "/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251028_150110/UX_nan_filtered.npy"
    uy_data_dir = "/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251028_150110/UY_nan_filtered.npy"
    mask_dir = "/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251028_150110/mask.npy"
    logger.info(f"Data paths: ux={ux_data_dir}, uy={uy_data_dir}")

    # 计时数据加载耗时
    begin_time = time.time()
    #traj应为包含时序流场信息的 NumPy 数组；加载 Ux 与 Uy 并组装为双通道
    traj_ux = np.load(ux_data_dir)
    traj_uy = np.load(uy_data_dir)
    traj_ux = np.expand_dims(traj_ux, axis=0)
    traj_uy = np.expand_dims(traj_uy, axis=0)
    #确保shape一样
    if traj_ux.shape != traj_uy.shape:
        logger.info(f"! Ux and Uy shapes differ: {traj_ux.shape} vs {traj_uy.shape}")
        sys.exit()
    traj = np.stack([traj_ux, traj_uy], axis=2)  # [N, T, C, nx, ny]

    #检查是否有异常的时间步
    bad_timesteps = []
    for t in range(traj.shape[1]):
        frame = traj[0, t]
        if np.abs(frame).max() > 10.0:
            logger.info(f"! Time step {t} has extreme values: max={np.abs(frame).max():.2e}")
            bad_timesteps.append(t)
    if bad_timesteps:
        logger.info(f"Bad time steps: {bad_timesteps}")
        sys.exit()

    logger.info(f"Mask path: {mask_dir}")


    # 加载二值掩码并应用到数据，屏蔽非流体区域
    mask = np.load(mask_dir).reshape(1, 1, traj.shape[-2], traj.shape[-1])
    traj = traj * mask
    logger.info(f"Data loading time: {time.time() - begin_time:.2f}s")

# 自适应按比例划分：训练80%，验证10%，测试10%，并确保验证/测试至少各有1帧
    nt_all = traj.shape[1]
    train_end = int(nt_all * 0.8)
    val_end = int(nt_all * 0.9)
    train_end = max(1, min(train_end, nt_all - 2))
    val_end = max(train_end + 1, min(val_end, nt_all - 1))
    traj_train = traj[:, :train_end]
    traj_val = traj[:, train_end:val_end]
    traj_test = traj[:, val_end:]

    logger.info(f"Shape of whole data (traj): {traj.shape}")
    logger.info(f"Shape of train data (traj_train): {traj_train.shape}")
    logger.info(f"Shape of val data (traj_val): {traj_val.shape}")
    logger.info(f"Shape of test data (traj_test): {traj_test.shape}\n")

##################初始化模型训练中的核心参数字典 Par##############################
    Par = {}
    Par["nx"] = traj_train.shape[-2]
    Par["ny"] = traj_train.shape[-1]
    Par["nf"] = 2 #特征数
    Par["d_emb"] = 128 #dimension of embedding" 的缩写，通常表示嵌入层的维度

    logger.info(f"Dimension of flow (nx*ny): ({Par['nx']}, {Par['ny']})")
    logger.info(f"Number of features (nf): {Par['nf']}")

    Par["lb"] = 10 #lookback 输入时间窗口长度
    Par["lf"] = 2 #lookforward 单次预测的时间窗口长度
    Par["LF"] = 10 #"long-term forecast" ，代表长序列预测的总长度（验证 / 测试阶段的目标预测长度）
    Par["channels"] = Par["nf"] * Par["lb"] # 输入通道数=特征数*输入时间窗口长度
    Par["num_epochs"] = 500
    logger.info(f"Number of timesteps as inputs (lb): {Par['lb']}")
    logger.info(f"Number of timesteps as outputs (lf): {Par['lf']}")
    logger.info(f"Number of timesteps for long-term prediction (LF): {Par['LF']}")
    logger.info(f"Number epochs: {Par['num_epochs']}\n")

    time_cond = np.linspace(0, 1, Par["lf"])
    if Par["lf"] == 1:
        time_cond = np.linspace(0, 1, Par["lf"]) + 1

    t_min = np.min(time_cond)
    t_max = np.max(time_cond)
    if Par["lf"] == 1:
        t_min = 0
        t_max = 1

    # 用于对模型输入数据（历史流场数据）进行标准化处理，
    # 公式为：输入数据标准化后 = (原始输入 - inp_shift) / inp_scale。
    Par["inp_shift"] = float(np.mean(traj_train))
    Par["out_shift"] = float(np.mean(traj_train))
    inp_std = float(np.std(traj_train))
    out_std = float(np.std(traj_train))
    Par["inp_scale"] = float(max(inp_std, 1e-6))
    Par["out_scale"] = float(max(out_std, 1e-6))
    # 时间条件归一化后 = (原始时间条件 - t_shift) / t_scale。
    Par["t_shift"] = float(t_min)
    Par["t_scale"] = float(t_max - t_min)
    Par["time_cond"] = time_cond.tolist()
    
    # 添加物理约束的权重参数
    Par["mse_weight"] = 1.0  # MSE损失的权重
    Par["div_weight"] = 0.1  # 连续性方程约束的权重
    Par["momentum_weight"] = 0.05  # 动量方程约束的权重
    
    logger.info(f"Input shift of trai_train: {Par['inp_shift']}")
    logger.info(f"Input scale of trai_train: {Par['inp_scale']}")
    logger.info(f"Output shift of trai_train: {Par['out_shift']}")
    logger.info(f"Output scale of trai_train: {Par['out_scale']}")
    logger.info(f"Time shift: {Par['t_shift']}")
    logger.info(f"Time scale: {Par['t_scale']}")
    logger.info(f"Time cond: {Par['time_cond']}")
    logger.info(f"Physical constraint weights - MSE: {Par['mse_weight']}, Div: {Par['div_weight']}, Momentum: {Par['momentum_weight']}\n")

    Par["mask"] = paddle.to_tensor(mask, dtype=data_type)

    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, paddle.Tensor)):
            return obj.tolist()
        return obj

    with open(os.path.join(main_output_dir, "Par.json"), "w") as f:
        json.dump(Par, f, default=convert_to_serializable)

    #将数据集转换为 PaddlePaddle 张量格式
    traj_train_tensor = paddle.to_tensor(data=traj_train, dtype=data_type)
    traj_val_tensor = paddle.to_tensor(data=traj_val, dtype=data_type)
    traj_test_tensor = paddle.to_tensor(data=traj_test, dtype=data_type)
    time_cond_tensor = paddle.to_tensor(data=time_cond, dtype=data_type)
    begin_time = time.time()
    x_idx_train, t_idx_train, y_idx_train = preprocess_train(traj_train, Par)
    logger.info("Shape of train dataset")
    logger.info(f"x_idx_train: {x_idx_train.shape}")
    logger.info(f"t_idx_train: {t_idx_train.shape}")
    logger.info(f"y_idx_train: {y_idx_train.shape}\n")
    x_idx_val, t_idx_val, y_idx_val = preprocess(traj_val, Par)
    logger.info("Shape of val dataset")
    logger.info(f"x_idx_val: {x_idx_val.shape}")
    logger.info(f"t_idx_val: {t_idx_val.shape}")
    logger.info(f"y_idx_val: {y_idx_val.shape}\n")
    x_idx_test, t_idx_test, y_idx_test = preprocess(traj_test, Par)
    logger.info("Shape of test dataset")
    logger.info(f"x_idx_test: {x_idx_test.shape}")
    logger.info(f"t_idx_test: {t_idx_test.shape}")
    logger.info(f"y_idx_test: {y_idx_test.shape}\n")
    logger.info(f"Data preprocess time: {time.time() - begin_time:.2f}s\n")
    train_dataset = YourDataset_train(x_idx_train, t_idx_train, y_idx_train)
    val_dataset = YourDataset(x_idx_val, y_idx_val)
    test_dataset = YourDataset(x_idx_test, y_idx_test)

    # 降低批大小以缓解显存压力（远端 GPU 出现 OOM）
    train_batch_size = 4  # 100
    val_batch_size = 4  # 100
    test_batch_size = 4  # 100
    logger.info(f"Batch size of train, val, and test: {train_batch_size}, {val_batch_size}, {test_batch_size}")
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = paddle.io.DataLoader(dataset=val_dataset, batch_size=val_batch_size)
    test_loader = paddle.io.DataLoader(dataset=test_dataset, batch_size=test_batch_size)
    # 适当降低注意力头数，进一步减少显存占用
    model = Unet2D_with_FNO(
        dim=16,
        Par=Par,
        dim_mults=(1, 2, 4, 8),
        channels=Par["channels"],
        attention_heads=2,
    ).astype("float32")

    # 初始化损失函数，传入机翼掩码
    criterion = CustomLoss(Par, mask=Par.get('mask'))
    # 初始化优化器
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=5 * 1e-04,
        weight_decay=1e-06,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
    )
    # 初始化学习率调度器
    scheduler = combined_scheduler(
        optimizer,
        Par["num_epochs"] * len(train_loader),
        int(0.1 * Par["num_epochs"]) * len(train_loader),
    )

    # Training loop
    num_epochs = Par["num_epochs"]
    PLOT_EVERY = 1 #绘图频率，每间隔PLOT_EVERY个epoch绘制一次图像
    best_val_loss = float("inf")
    best_model_id = 0
    # 记录每个 epoch 的损失以绘制曲线
    train_losses = []
    val_losses = []
    
    # 在主文件夹下创建models和images子文件夹
    models_dir = os.path.join(main_output_dir, "models")
    images_dir = os.path.join(main_output_dir, "images")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    # 初始化损失CSV并写入表头
    loss_csv_path = os.path.join(main_output_dir, "loss_curve.csv")
    try:
        with open(loss_csv_path, "w", encoding="utf8") as f:
            f.write("epoch,train_loss,val_loss\n")
    except Exception as e:
        logger.warning(f"Failed to init loss CSV: {e}")
    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        begin_time = time.time()
        model.train()
        train_loss = 0.0
        # 物理项 warmup：前 20% 轮次（至少 30 epoch）线性引入，降低早期非有限损失
        phys_warmup_epochs = max(30, int(0.2 * num_epochs))
        warmup_factor = min(1.0, epoch / float(phys_warmup_epochs))
        try:
            criterion.set_warmup_factor(warmup_factor)
        except Exception:
            pass
        for x_idx, t_idx, y_idx in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not sys.stdout.isatty(),
        ): 
            x = traj_train_tensor[0, x_idx]
            t = time_cond_tensor[t_idx]
            y_true = traj_train_tensor[0, y_idx]
            optimizer.clear_gradients(set_to_zero=False)

            # 关闭自动混合精度，仅用 GradScaler 缩放损失；如需开启混合精度请将 enable=True。
            with paddle.amp.auto_cast(enable=False):
                y_pred = model(x, t)
            # 在 FP32 中计算损失，提升数值稳定性
            loss = criterion(y_pred.astype('float32'), y_true.astype('float32'))
            # 若出现非有限值则跳过该批，避免 GradScaler 降到 0
            if not np.isfinite(float(loss.item())):
                # 尝试一次数值修复：替换预测的 NaN/Inf，并暂时关闭物理项
                y_pred_safe = paddle.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    criterion.set_warmup_factor(0.0)
                except Exception:
                    pass
                loss_fallback = criterion(y_pred_safe.astype('float32'), y_true.astype('float32'))
                if not np.isfinite(float(loss_fallback.item())):
                    logger.warning("Non-finite loss detected; skipping this batch")
                    optimizer.clear_gradients(set_to_zero=True)
                    continue
                else:
                    loss = loss_fallback

            loss.backward()
            try:
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(paddle.flatten(p.grad))
                if grads:
                    total_norm = float(paddle.linalg.norm(paddle.concat(grads), p=2).numpy())
                    if not np.isfinite(total_norm):
                        logger.warning("Non-finite grad norm detected; skipping optimizer update for this batch")
                        optimizer.clear_gradients(set_to_zero=True)
                        continue
                    if epoch <= 5:
                        logger.info(f"Grad norm: {total_norm:.3e}")
            except Exception:
                pass
            optimizer.step()
            train_loss += loss.item()
            # 每个训练批次更新学习率（warmup + 余弦退火）。
            scheduler.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with paddle.no_grad():
            for x_idx, y_idx in val_loader:
                x = traj_val_tensor[0, x_idx]
                t = time_cond_tensor[t_idx_val]
                y_true = traj_val_tensor[0, y_idx]
                NT = Par["lb"] + y_true.shape[1]  # 动态设置 NT 以匹配当前窗口长度
                y_pred = rollout(model, x, t, NT, Par, val_batch_size)
                # 关闭自动混合精度，仅用 GradScaler 缩放损失；如需开启混合精度请将 enable=True。
                loss = criterion(y_pred.astype('float32'), y_true.astype('float32'))
                val_loss += loss.item()
            # 保存本轮的真值/预测对比与频谱图到 images/。
            val_loss_avg = val_loss / len(val_loader) if len(val_loader) > 0 else float('nan')
            val_losses.append(val_loss_avg)
            if (epoch == 1) or (epoch == num_epochs) or (epoch % PLOT_EVERY == 0):
                make_plot(
                    y_true.detach().cpu().numpy(),
                    y_pred.detach().cpu().numpy(),
                    epoch,
                    images_dir,
                    train_losses=train_losses,
                    val_losses=val_losses,
                )
        val_loss = val_loss_avg
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_id = epoch
            paddle.save(obj=model.state_dict(), path=os.path.join(models_dir, "best_model.pdparams"))
        elapsed_time = time.time() - begin_time
        logger.info(
            f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, "
            f"Best model: {best_model_id}, Learning rate: {scheduler.get_lr():.4e}, "
            f"Epoch time: {elapsed_time:.2f}"
        )
        # 每个epoch结束后追加一行到CSV
        try:
            with open(loss_csv_path, "a", encoding="utf8") as f:
                f.write(f"{epoch},{train_loss:.6e},{val_loss:.6e}\n")
        except Exception as e:
            logger.warning(f"Failed to append loss CSV: {e}")

        
    logger.info("Training finished.")
    logger.info(f"Training Time: {time.time() - t0:.1f}s")
    
    # Loss CSV is appended per epoch; end-of-training write removed.

    model.eval()
    test_loss = 0.06
    with paddle.no_grad():
        for x_idx, y_idx in test_loader:
            x = traj_test_tensor[0, x_idx]
            t = time_cond_tensor[t_idx_test]
            y_true = traj_test_tensor[0, y_idx]
            NT = Par["lb"] + y_true.shape[1]  # 动态设置 NT
            y_pred = rollout(model, x, t, NT, Par, val_batch_size)
            # 关闭自动混合精度，仅用 GradScaler 缩放损失；如需开启混合精度请将 enable=True。
            # 测试阶段同样在 FP32 中计算损失，避免数值问题
            loss = criterion(y_pred.astype('float32'), y_true.astype('float32'))
            test_loss += loss.item()
    test_loss /= len(test_loader)
    logger.info(f"Test Loss: {test_loss:.4e}")
