import json
import logging
import math
import os
import random
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 在导入 paddle 之前处理设备选择：支持 "--devices" 或 "-d" 指定 GPU，如 "0"、"1"、"0,1"
try:
    # 清理旧的限制性环境变量
    if 'FLAGS_selected_gpus' in os.environ:
        del os.environ['FLAGS_selected_gpus']

    # 轻量解析命令行以读取 --devices/-d（必须在导入 paddle 之前）
    devices_arg = None
    for i, arg in enumerate(sys.argv):
        if arg in ('--devices', '-d'):
            if i + 1 < len(sys.argv):
                devices_arg = sys.argv[i + 1]
            break

    if devices_arg and isinstance(devices_arg, str) and len(devices_arg) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = devices_arg
    else:
        # 若未显式设置，默认仅暴露单卡 0
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paddle
import scipy.stats as stats
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from model.unet_withFNO import Unet2D_with_FNO
# 优先从项目工具包导入单进程 DataParallel 封装；若不存在则使用本地 fallback
try:
    from ppcfd.utils.parallel import setup_singleprocess_dp
except Exception:
    def setup_singleprocess_dp(model, optimizer=None):
        try:
            gpu_count = 0
            try:
                gpu_count = paddle.device.cuda.device_count()
            except Exception:
                gpu_count = 0
            if gpu_count >= 1:
                model = paddle.DataParallel(model)
            return model, optimizer
        except Exception:
            return model, optimizer

# 诊断开关：关闭分阶段损失打印，减少训练日志噪声
LOSS_DIAG = False


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
    try:
        gpu_available = bool(paddle.is_compiled_with_cuda())
        try:
            gpu_count = paddle.device.cuda.device_count()
        except Exception:
            gpu_count = 0
        device = "gpu" if gpu_available and gpu_count > 0 else "cpu"
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
    try:
        logger.info(f"Env CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '')}")
        logger.info(f"Env LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    except Exception:
        pass

    # 单进程模式：不初始化分布式环境，避免与 DataParallel 冲突
    return logger


def make_plot(TRUE, PRED, epoch, images_dir="images", train_losses=None, val_losses=None, val_rollout_losses=None, val_1step_losses=None, mask2d=None, zero_center=True, unify_scale=False):
    """
    可视化验证结果：
    - 每列对应一个被选时间点，按该列帧的值域自适应设定色阶，避免整片绿色。
    - 计算并在标题显示相对 L2 误差（返回 float）。
    - 进行频谱对比：2D FFT -> 能量谱 -> 分箱 -> 对数坐标绘制，可叠加 `k^-5/3` 参考线。
    - 每个 epoch 保存图像到指定的 images 目录。
    """
    sample_id = 0
    T = TRUE.shape[1]
    # 依据真实时间长度选点，最多展示 3 列
    skip_t = max(1, T // 8)  # 适度下采样，保证覆盖不同时间
    idx_all = np.arange(T)[::skip_t]
    N = min(idx_all.shape[0], 3)
    idx = idx_all[:N]

    # 只展示 Ux（第 0 通道）；如需 Uy 把 0 改为 1
    true = np.nan_to_num(TRUE[sample_id, idx, 0], nan=0.0, posinf=0.0, neginf=0.0)
    pred1 = np.nan_to_num(PRED[sample_id, idx, 0], nan=0.0, posinf=0.0, neginf=0.0)
    # 同步提取 Uy 通道
    true_uy = np.nan_to_num(TRUE[sample_id, idx, 1], nan=0.0, posinf=0.0, neginf=0.0)
    pred_uy = np.nan_to_num(PRED[sample_id, idx, 1], nan=0.0, posinf=0.0, neginf=0.0)
    time_ls = idx
    CMAP = "turbo"

    def rel_l2(Ta, Pa):
        # 安全获取当前默认 dtype，若不可用则回退到 float32
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

    def get_vmin_vmax(a, b):
        # 过滤掉无效/近零背景，避免掩膜主导色阶
        def valid_vals(x):
            vals = x[np.isfinite(x)]
            vals = vals[np.abs(vals) > 1e-8]
            # 如果有效像素过少，退回到全部像素
            if vals.size < 50:
                vals = x.reshape(-1)
            return vals

        va = valid_vals(a)
        vb = valid_vals(b)
        stack = np.concatenate([va, vb]) if va.size and vb.size else np.concatenate([va, vb])

        if stack.size == 0:
            # 极端退路：用两帧的最小/最大
            vmin = float(np.nanmin(np.stack([a, b])))
            vmax = float(np.nanmax(np.stack([a, b])))
            return vmin, vmax

        # 首选：百分位自适应
        vmin = np.percentile(stack, 1)
        vmax = np.percentile(stack, 99)

        # 备用：均值±3σ 提升对比度
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            m = float(stack.mean())
            s = float(stack.std())
            if s > 0:
                vmin = m - 3.0 * s
                vmax = m + 3.0 * s
            else:
                vmin = float(stack.min())
                vmax = float(stack.max())
        return float(vmin), float(vmax)

    def frame_vmin_vmax(x):
        # 单帧鲁棒色阶：忽略近零与无效值，百分位自适应
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

    # 准备可选掩膜（二维）用于显示时一致处理真值/预测
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
        # 选项：零中心色阶，使色卡一定包含 0
        if zero_center:
            max_abs_t = max(abs(vmin_t), abs(vmax_t))
            max_abs_p = max(abs(vmin_p), abs(vmax_p))
            max_abs_t = max(max_abs_t, 1e-12)
            max_abs_p = max(max_abs_p, 1e-12)
            vmin_t, vmax_t = -max_abs_t, max_abs_t
            vmin_p, vmax_p = -max_abs_p, max_abs_p
        # 选项：真值与预测共用同一色阶，便于横向对比
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

        # 频谱：Ux 真值和预测值对比
        # 真值频谱
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
        
        # 预测值频谱
        image = pred1[i]
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
        Abins_pred, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins_pred *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_pred = np.maximum(Abins_pred, 1e-12)
        axes[2, i].loglog(kvals, Abins_pred, 'r--', label="Pred Ux", linewidth=2)
        
        # 理论参考线
        k_ref = np.linspace(1, np.max(kbins), 100)
        energy_ref = k_ref ** (-5 / 3)
        energy_ref *= max(max(Abins_true), max(Abins_pred)) / max(energy_ref)
        axes[2, i].loglog(k_ref, energy_ref, "k:", label="k^-5/3", alpha=0.7)
        axes[2, i].legend(fontsize=9)
        axes[2, i].set_xlabel("$k$")
        axes[2, i].set_title("Ux Energy Spectrum", fontsize=12)

        # Uy 通道：图像（独立色阶，可选掩膜；支持零中心/共用色阶）
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

        # 频谱：Uy 真值和预测值对比
        # 计算真值频谱
        image = true_uy[i]
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
        
        # 计算预测值频谱
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
        
        # 绘制对比图
        axes[5, i].loglog(kvals, Abins_true, label="Simulated Uy", color="tab:blue")
        axes[5, i].loglog(kvals, Abins_pred, label="MATCHO Uy", color="tab:orange")
        
        # 添加理论参考线
        k_ref = np.linspace(1, np.max(kbins), 100)
        energy_ref = k_ref ** (-5 / 3)
        energy_ref *= max(max(Abins_true), max(Abins_pred)) / max(energy_ref)
        axes[5, i].loglog(k_ref, energy_ref, "k--", label="k^-5/3")
        
        axes[5, i].legend(fontsize=10)
        axes[5, i].set_title(f"Uy Energy Spectrum (t={int(time_ls[i])})")
        axes[5, i].set_xlabel("$k$")
        axes[5, i].set_ylabel("Energy")
    # 预留底部空间给损失曲线，避免与子图重叠
    fig.subplots_adjust(left=0.06, right=0.9, top=0.96, bottom=0.22, hspace=0.8, wspace=0.25)
    # 兼容：若传入旧参数 val_losses，则视为 rollout 损失
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
            # 将非有限值置为 NaN，便于绘图和尺度判断
            train_arr = np.where(np.isfinite(train_arr), train_arr, np.nan)
            val_arr = np.where(np.isfinite(val_arr), val_arr, np.nan)
            if val1_arr is not None:
                val1_arr = np.where(np.isfinite(val1_arr), val1_arr, np.nan)

            # 判断是否存在正值；若不存在则不使用 log 尺度
            has_train_pos = np.any((train_arr > 0) & np.isfinite(train_arr))
            has_val_pos = np.any((val_arr > 0) & np.isfinite(val_arr))
            has_val1_pos = np.any((val1_arr > 0) & np.isfinite(val1_arr)) if val1_arr is not None else False
            use_log = bool(has_train_pos or has_val_pos or has_val1_pos)

            if use_log:
                loss_ax.set_yscale("log")
                # 仅绘制正值，避免 log 尺度下的非正值导致的警告
                train_plot = np.where(train_arr > 0, train_arr, np.nan)
                val_plot = np.where(val_arr > 0, val_arr, np.nan)
                loss_ax.plot(x, train_plot, label="Train loss", color="tab:blue", linewidth=2)
                loss_ax.plot(val_x, val_plot, label="Val rollout loss", color="tab:orange", linestyle="--", marker="o", markersize=3, alpha=0.9, linewidth=2, zorder=3)
                if val1_arr is not None:
                    val1_x = np.arange(1, len(val1_arr) + 1)
                    val1_plot = np.where(val1_arr > 0, val1_arr, np.nan)
                    loss_ax.plot(val1_x, val1_plot, label="Val 1-step loss", color="tab:green", linestyle=":", marker="s", markersize=3, alpha=0.9, linewidth=2, zorder=2)
                # 设置正值范围的 y 轴上下界，保证 log 尺度有效
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
                # 无正值时使用线性尺度
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
    def __init__(self, Par, mask=None, logger=None):
        super(CustomLoss, self).__init__()
        self.Par = Par
        # 添加物理约束的权重参数
        self.mse_weight = Par.get('mse_weight', 1.0)
        self.div_weight = Par.get('div_weight', 0.1)  # 连续性方程权重
        self.momentum_weight = Par.get('momentum_weight', 0.05)  # 动量方程权重
        # 物理项 warmup 因子（0~1），训练早期降低物理项权重以稳定损失
        self.warmup_factor = 1.0
        # 流体参数
        self.Ma = 0.3  # 马赫数
        self.Re = 23000  # 雷诺数
        self.nu = 1.0 / self.Re  # 运动粘度
        # 机翼掩码，用于排除机翼区域
        self.mask = mask
        # 分阶段诊断
        self.enable_diag = LOSS_DIAG
        # 注入 logger，兼容多进程；若为空则使用根 logger
        self.logger = logger if logger is not None else logging.getLogger("")

    def _diag_check(self, name: str, tensor: paddle.Tensor, verbose: bool = False) -> bool:
        """检查张量是否全为有限值，并在需要时打印统计信息。
        - 当张量包含非有限值时，打印 WARNING 级别日志；
        - 当开启 LOSS_DIAG 且 verbose=True 时，打印 INFO 级别统计日志。
        返回：True 表示全为有限，False 表示存在 NaN/Inf。
        """
        try:
            t = tensor.astype('float32') if tensor.dtype in [paddle.float16, paddle.bfloat16] else tensor
            # 统计只在必要时进行，避免过多开销
            finite_mask = paddle.isfinite(t)
            is_all_finite = bool(finite_mask.all().item())
            if (not is_all_finite) or (self.enable_diag and verbose):
                # 计算基础统计量（对 NaN 安全）
                total = int(np.prod(t.shape)) if len(t.shape) > 0 else 1
                try:
                    min_v = float(paddle.nanmin(t).item()) if total > 0 else 0.0
                    max_v = float(paddle.nanmax(t).item()) if total > 0 else 0.0
                    mean_v = float(paddle.nanmean(t).item()) if total > 0 else 0.0
                except Exception:
                    # 某些 Paddle 版本可能不支持 nanmin/nanmean；退化为普通统计
                    min_v = float(paddle.min(t).item()) if total > 0 else 0.0
                    max_v = float(paddle.max(t).item()) if total > 0 else 0.0
                    mean_v = float(paddle.mean(t).item()) if total > 0 else 0.0
                try:
                    non_finite = int((~finite_mask).sum().item())
                except Exception:
                    non_finite = 0
                level = logging.WARNING if not is_all_finite else logging.INFO
                self.logger.log(
                    level,
                    f"[LOSS-DIAG] {name}: finite={is_all_finite}, non_finite={non_finite}/{total}, "
                    f"min={min_v:.3e}, max={max_v:.3e}, mean={mean_v:.3e}, shape={list(t.shape)}, dtype={tensor.dtype}",
                )
            return is_all_finite
        except Exception as e:
            self.logger.warning(f"[LOSS-DIAG] {name}: check failed: {e}")
            return False

    def set_warmup_factor(self, factor: float):
        # 将 warmup 因子限定在 [0,1]
        try:
            self.warmup_factor = float(max(0.0, min(1.0, factor)))
        except Exception:
            self.warmup_factor = 1.0

    def forward(self, y_pred, y_true):
        # 标准化到训练期的输出分布（只用流体区域统计得到的 shift/scale）
        y_true = (y_true - self.Par["out_shift"]) / (self.Par["out_scale"])
        y_pred = (y_pred - self.Par["out_shift"]) / (self.Par["out_scale"])
        # 诊断用的钳制副本（不参与反传）
        try:
            y_true_diag = paddle.clip(y_true.detach().astype('float32'), min=-10.0, max=10.0)
            y_pred_diag = paddle.clip(y_pred.detach().astype('float32'), min=-10.0, max=10.0)
        except Exception:
            y_true_diag, y_pred_diag = y_true.detach(), y_pred.detach()

        # 诊断：输入规范化后的张量
        if self.enable_diag:
            self._diag_check("y_true_norm", y_true_diag, verbose=True)
            self._diag_check("y_pred_norm", y_pred_diag, verbose=True)
        
        # 1. MSE损失 - 只计算流体区域（使用掩码）
        if self.mask is not None:
            # 创建与预测结果匹配的掩码
            mask_tensor = paddle.to_tensor(self.mask, dtype=y_pred.dtype)
            # 扩展掩码维度以匹配y_pred的维度
            while len(mask_tensor.shape) < len(y_pred.shape):
                mask_tensor = paddle.unsqueeze(mask_tensor, axis=0)
            # 计算掩码区域的MSE损失
            squared_diff = (y_true - y_pred) ** 2
            masked_squared_diff = squared_diff * mask_tensor
            mse_loss = paddle.sum(masked_squared_diff) / (paddle.sum(mask_tensor) + paddle.to_tensor(1e-12, dtype=masked_squared_diff.dtype))
            if self.enable_diag:
                ok_mse = self._diag_check("mse_masked_squared_diff", masked_squared_diff)
                if not ok_mse:
                    # 进一步展开诊断
                    self._diag_check("mse_squared_diff", squared_diff)
                    self._diag_check("mse_mask_tensor", mask_tensor)
        else:
            mse_loss = paddle.mean((y_true - y_pred) ** 2)
            if self.enable_diag:
                self._diag_check("mse_squared_diff", (y_true - y_pred) ** 2)

        # 2. 物理约束惩罚项
        # 可选的空间下采样以降低梯度计算开销
        phys_stride = int(self.Par.get('phys_stride', 1))
        def _downsample_spatial(t: paddle.Tensor, s: int):
            if s is None or s <= 1:
                return t
            # 统一认为最后两个维度为空间维（nx, ny），兼容4D/5D布局
            try:
                return t[..., ::s, ::s]
            except Exception:
                return t
        y_phys = _downsample_spatial(y_pred, phys_stride)
        physics_loss = 0.0
        
        # 计算速度场的散度作为连续性方程约束（可压缩流体近似）
        div_w = float(self.div_weight) * float(self.warmup_factor)
        if div_w > 0:
            # 计算ux对x的偏导和uy对y的偏导
            # 兼容两种布局：
            # - 4D: [B, C(=2), nx, ny] 或 [B, nx, ny, C(=2)]
            # - 5D: [B, T, C(=2), nx, ny] 或 [B, T, nx, ny, C(=2)]
            if y_phys.ndim == 4:
                if y_phys.shape[1] == 2:
                    ux = y_phys[:, 0, :, :]
                    uy = y_phys[:, 1, :, :]
                elif y_phys.shape[-1] == 2:
                    ux = y_phys[..., 0]
                    uy = y_phys[..., 1]
                else:
                    raise ValueError(f"y_pred 形状不符合期望的4D布局: {y_phys.shape}")
            elif y_phys.ndim == 5:
                if y_phys.shape[2] == 2:
                    ux = y_phys[:, :, 0, :, :]
                    uy = y_phys[:, :, 1, :, :]
                elif y_phys.shape[-1] == 2:
                    ux = y_phys[..., 0]
                    uy = y_phys[..., 1]
                else:
                    raise ValueError(f"y_pred 形状不符合期望的5D布局: {y_phys.shape}")
            else:
                raise ValueError(f"y_pred 维度不支持: ndim={y_phys.ndim}, shape={y_phys.shape}")
            
            # 使用中心差分近似，不使用周期边界条件
            # 针对极小尺寸(宽/高为1或2)做健壮处理，避免切片赋值形状为0造成错误
            def diff_x(field):
                """对最后一维（宽度）做一阶差分，避免切片赋值导致空切片错误"""
                w = int(field.shape[-1])
                if w >= 3:
                    center = (field[..., 2:] - field[..., :-2]) * 0.5
                    left = (field[..., 1] - field[..., 0]).unsqueeze(axis=-1)
                    right = (field[..., -1] - field[..., -2]).unsqueeze(axis=-1)
                    return paddle.concat([left, center, right], axis=-1)
                elif w == 2:
                    diff = (field[..., 1] - field[..., 0]).unsqueeze(axis=-1)
                    return paddle.concat([diff, diff], axis=-1)
                else:  # w == 1 或异常
                    return paddle.zeros_like(field)

            def diff_y(field):
                """对倒数第二维（高度）做一阶差分，避免切片赋值导致空切片错误"""
                h = int(field.shape[-2])
                if h >= 3:
                    center = (field[..., 2:, :] - field[..., :-2, :]) * 0.5
                    top = (field[..., 1, :] - field[..., 0, :]).unsqueeze(axis=-2)
                    bottom = (field[..., -1, :] - field[..., -2, :]).unsqueeze(axis=-2)
                    return paddle.concat([top, center, bottom], axis=-2)
                elif h == 2:
                    diff = (field[..., 1, :] - field[..., 0, :]).unsqueeze(axis=-2)
                    return paddle.concat([diff, diff], axis=-2)
                else:  # h == 1 或异常
                    return paddle.zeros_like(field)

            # 对x方向求导（宽度方向）
            dudx = diff_x(ux)
            # 对y方向求导（高度方向）
            dvdy = diff_y(uy)
            
            # 计算散度
            divergence = dudx + dvdy

            # 诊断用的钳制副本（不参与反传）
            try:
                divergence_diag = paddle.clip(divergence.detach(), min=-10.0, max=10.0)
            except Exception:
                divergence_diag = divergence.detach()

            if self.enable_diag:
                ok_div = self._diag_check("divergence", divergence_diag)
                if not ok_div:
                    self._diag_check("dudx", dudx)
                    self._diag_check("dvdy", dvdy)

            # 使用掩码排除机翼区域
            if self.mask is not None:
                mask_tensor = paddle.to_tensor(self.mask, dtype=divergence.dtype)
                while len(mask_tensor.shape) < len(divergence.shape):
                    mask_tensor = paddle.unsqueeze(mask_tensor, axis=0)
                if phys_stride > 1:
                    mask_tensor = mask_tensor[..., ::phys_stride, ::phys_stride]
                masked_divergence = divergence * mask_tensor
                den = paddle.sum(mask_tensor) + paddle.to_tensor(1e-12, dtype=masked_divergence.dtype)
                div_loss = paddle.sum(masked_divergence ** 2) / den
            else:
                div_loss = paddle.mean(divergence ** 2)

            if self.enable_diag:
                self._diag_check("div_loss", div_loss)
            
            physics_loss += div_w * div_loss
        
        # 3. 简化的动量方程约束（忽略压力项）
        mom_w = float(self.momentum_weight) * float(self.warmup_factor)
        if mom_w > 0:
            # 与散度项保持一致的通道提取逻辑，兼容4D/5D与通道末尾布局
            if y_phys.ndim == 4:
                if y_phys.shape[1] == 2:
                    ux = y_phys[:, 0, :, :]
                    uy = y_phys[:, 1, :, :]
                elif y_phys.shape[-1] == 2:
                    ux = y_phys[..., 0]
                    uy = y_phys[..., 1]
                else:
                    raise ValueError(f"y_pred 形状不符合期望的4D布局: {y_phys.shape}")
            elif y_phys.ndim == 5:
                if y_phys.shape[2] == 2:
                    ux = y_phys[:, :, 0, :, :]
                    uy = y_phys[:, :, 1, :, :]
                elif y_phys.shape[-1] == 2:
                    ux = y_phys[..., 0]
                    uy = y_phys[..., 1]
                else:
                    raise ValueError(f"y_pred 形状不符合期望的5D布局: {y_phys.shape}")
            else:
                raise ValueError(f"y_pred 维度不支持: ndim={y_phys.ndim}, shape={y_phys.shape}")

            # 稳健的一阶差分（与上面的连续性约束一致的实现）
            def diff_x(field):
                """与上方一致的稳健一阶差分（宽度维），用于动量项"""
                w = int(field.shape[-1])
                if w >= 3:
                    center = (field[..., 2:] - field[..., :-2]) / 2.0
                    left = (field[..., 1] - field[..., 0]).unsqueeze(axis=-1)
                    right = (field[..., -1] - field[..., -2]).unsqueeze(axis=-1)
                    return paddle.concat([left, center, right], axis=-1)
                elif w == 2:
                    diff = (field[..., 1] - field[..., 0]).unsqueeze(axis=-1)
                    return paddle.concat([diff, diff], axis=-1)
                else:
                    return paddle.zeros_like(field)

            def diff_y(field):
                """与上方一致的稳健一阶差分（高度维），用于动量项"""
                h = int(field.shape[-2])
                if h >= 3:
                    center = (field[..., 2:, :] - field[..., :-2, :]) / 2.0
                    top = (field[..., 1, :] - field[..., 0, :]).unsqueeze(axis=-2)
                    bottom = (field[..., -1, :] - field[..., -2, :]).unsqueeze(axis=-2)
                    return paddle.concat([top, center, bottom], axis=-2)
                elif h == 2:
                    diff = (field[..., 1, :] - field[..., 0, :]).unsqueeze(axis=-2)
                    return paddle.concat([diff, diff], axis=-2)
                else:
                    return paddle.zeros_like(field)

            # 稳健的二阶差分（用于粘性项）
            def diff2_x(field):
                w = field.shape[-1]
                out = paddle.zeros_like(field)
                if w >= 3:
                    out[..., 1:-1] = field[..., 2:] - 2 * field[..., 1:-1] + field[..., :-2]
                    out[..., 0] = field[..., 1] - 2 * field[..., 0] + field[..., 0]
                    out[..., -1] = field[..., -1] - 2 * field[..., -1] + field[..., -2]
                elif w == 2:
                    # 近似为常值或零（不足以稳定估计二阶项）
                    out = out * 0.0
                else:
                    out = out * 0.0
                return out

            def diff2_y(field):
                h = field.shape[-2]
                out = paddle.zeros_like(field)
                if h >= 3:
                    out[..., 1:-1, :] = field[..., 2:, :] - 2 * field[..., 1:-1, :] + field[..., :-2, :]
                    out[..., 0, :] = field[..., 1, :] - 2 * field[..., 0, :] + field[..., 0, :]
                    out[..., -1, :] = field[..., -1, :] - 2 * field[..., -1, :] + field[..., -2, :]
                elif h == 2:
                    out = out * 0.0
                else:
                    out = out * 0.0
                return out

            # 一阶偏导
            dudx = diff_x(ux)
            dudy = diff_y(ux)
            dvdx = diff_x(uy)
            dvdy = diff_y(uy)

            # 二阶偏导（拉普拉斯项）
            d2udx2 = diff2_x(ux)
            d2udy2 = diff2_y(ux)
            d2vdx2 = diff2_x(uy)
            d2vdy2 = diff2_y(uy)

            # 对流项：u·∇u
            convective_x = ux * dudx + uy * dudy
            convective_y = ux * dvdx + uy * dvdy
            
            # 粘性项：ν∇²u
            viscous_x = self.nu * (d2udx2 + d2udy2)
            viscous_y = self.nu * (d2vdx2 + d2vdy2)
            
            # 动量方程残差（忽略压力项）
            momentum_residual_x = convective_x - viscous_x
            momentum_residual_y = convective_y - viscous_y

            # 诊断用的钳制副本（不参与反传）
            try:
                momentum_residual_x_diag = paddle.clip(momentum_residual_x.detach(), min=-10.0, max=10.0)
                momentum_residual_y_diag = paddle.clip(momentum_residual_y.detach(), min=-10.0, max=10.0)
            except Exception:
                momentum_residual_x_diag = momentum_residual_x.detach()
                momentum_residual_y_diag = momentum_residual_y.detach()

            if self.enable_diag:
                ok_mom = self._diag_check("momentum_residual_x", momentum_residual_x_diag) & self._diag_check("momentum_residual_y", momentum_residual_y_diag)
                if not ok_mom:
                    # 进一步细分来源
                    self._diag_check("convective_x", convective_x)
                    self._diag_check("convective_y", convective_y)
                    self._diag_check("viscous_x", viscous_x)
                    self._diag_check("viscous_y", viscous_y)
                    self._diag_check("dudx", dudx)
                    self._diag_check("dudy", dudy)
                    self._diag_check("dvdx", dvdx)
                    self._diag_check("dvdy", dvdy)
                    self._diag_check("d2udx2", d2udx2)
                    self._diag_check("d2udy2", d2udy2)
                    self._diag_check("d2vdx2", d2vdx2)
                    self._diag_check("d2vdy2", d2vdy2)
            
            # 组合动量方程残差
            momentum_residual = momentum_residual_x ** 2 + momentum_residual_y ** 2
            
            # 使用掩码排除机翼区域
            if self.mask is not None:
                mask_tensor = paddle.to_tensor(self.mask, dtype=momentum_residual.dtype)
                while len(mask_tensor.shape) < len(momentum_residual.shape):
                    mask_tensor = paddle.unsqueeze(mask_tensor, axis=0)
                if phys_stride > 1:
                    mask_tensor = mask_tensor[..., ::phys_stride, ::phys_stride]
                masked_momentum = momentum_residual * mask_tensor
                den = paddle.sum(mask_tensor) + paddle.to_tensor(1e-12, dtype=masked_momentum.dtype)
                momentum_loss = paddle.sum(masked_momentum) / den
            else:
                momentum_loss = paddle.mean(momentum_residual)

            if self.enable_diag:
                self._diag_check("momentum_loss", momentum_loss)
            
            physics_loss += mom_w * momentum_loss
        
        # 总损失
        total_loss = self.mse_weight * mse_loss + physics_loss

        # 可选：记录各个损失分量，便于调试
        if hasattr(self, 'loss_components'):
            self.loss_components = {
                'mse_loss': mse_loss.numpy(),
                'physics_loss': physics_loss.numpy()
            }

        if self.enable_diag:
            # 最后检查总损失
            self._diag_check("total_loss", total_loss, verbose=True)
        
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
        paddle.to_tensor(data=x_idx, dtype="int64", place=paddle.CPUPlace()),
        paddle.to_tensor(data=t_idx, dtype="int64", place=paddle.CPUPlace()),
        paddle.to_tensor(data=y_idx, dtype="int64", place=paddle.CPUPlace()),
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
        paddle.to_tensor(data=x_idx, dtype="int64", place=paddle.CPUPlace()),
        paddle.to_tensor(data=t_idx, dtype="int64", place=paddle.CPUPlace()),
        paddle.to_tensor(data=y_idx, dtype="int64", place=paddle.CPUPlace()),
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

def main_worker(rank: int = None):

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

    # —— 单卡模式：禁用分布式 ——
    world_size = 1
    rank_id = 0
    is_distributed = False
    is_main = True
    try:
        logger.info(f"[Single GPU] world_size={world_size}, rank={rank_id}, is_main={is_main}")
    except Exception:
        pass

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
    # 清理潜在的 NaN/Inf，防止后续损失出现非有限值
    traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
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
    # 只用流体区域计算统计量，避免掩膜零值将标准差压得过小
    mask_bool = (mask > 0.5)  # [1,1,nx,ny]
    # 广播到与 traj_train 相同的形状 [N, T, C, nx, ny]，避免布尔索引维度不匹配
    mask_5d = np.broadcast_to(mask_bool.reshape(1, 1, 1, Par["nx"], Par["ny"]), traj_train.shape)
    fluid_vals = traj_train[mask_5d]
    # 如果全是零（异常掩膜），回退到整体统计
    if fluid_vals.size == 0 or np.all(fluid_vals == 0):
        base_mean = float(np.mean(traj_train))
        base_std = float(np.std(traj_train))
    else:
        base_mean = float(np.mean(fluid_vals))
        base_std = float(np.std(fluid_vals))
        # 使用分位数的范围增强稳健性，防止异常值影响 scale
        try:
            p1, p99 = np.percentile(fluid_vals, [1.0, 99.0])
            robust_span = float(p99 - p1)
            base_std = float(max(base_std, robust_span / 2.0))
        except Exception:
            pass
    Par["inp_shift"] = base_mean
    Par["out_shift"] = base_mean
    Par["inp_scale"] = float(max(base_std, 1e-3))
    Par["out_scale"] = float(max(base_std, 1e-3))
    # 时间条件归一化后 = (原始时间条件 - t_shift) / t_scale。
    Par["t_shift"] = float(t_min)
    Par["t_scale"] = float(t_max - t_min)
    Par["time_cond"] = time_cond.tolist()
    
    # 添加物理约束的权重参数
    Par["mse_weight"] = 1.0  # MSE损失的权重
    Par["div_weight"] = 0.05  # 连续性方程约束的权重（降低初始权重，配合 warmup）
    Par["momentum_weight"] = 0.02  # 动量方程约束的权重（降低初始权重，配合 warmup）
    
    logger.info(f"Input shift of trai_train: {Par['inp_shift']}")
    logger.info(f"Input scale of trai_train: {Par['inp_scale']}")
    logger.info(f"Output shift of trai_train: {Par['out_shift']}")
    logger.info(f"Output scale of trai_train: {Par['out_scale']}")
    logger.info(f"Time shift: {Par['t_shift']}")
    logger.info(f"Time scale: {Par['t_scale']}")
    logger.info(f"Time cond: {Par['time_cond']}")
    logger.info(f"Physical constraint weights - MSE: {Par['mse_weight']}, Div: {Par['div_weight']}, Momentum: {Par['momentum_weight']}\n")

    Par["mask"] = paddle.to_tensor(mask, dtype=data_type)
    # 默认物理项下采样，降低梯度计算内存与耗时
    try:
        Par["phys_stride"] = int(Par.get("phys_stride", 2))
    except Exception:
        Par["phys_stride"] = 2

    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, paddle.Tensor)):
            return obj.tolist()
        return obj

    if is_main:
        with open(os.path.join(main_output_dir, "Par.json"), "w") as f:
            json.dump(Par, f, default=convert_to_serializable)

    # 将数据集转换为 PaddlePaddle 张量格式（在当前设备上创建，单卡模式避免每批次的 CPU->GPU 传输）
    traj_train_tensor = paddle.to_tensor(data=traj_train, dtype=data_type)
    traj_val_tensor = paddle.to_tensor(data=traj_val, dtype=data_type)
    traj_test_tensor = paddle.to_tensor(data=traj_test, dtype=data_type)
    time_cond_tensor = paddle.to_tensor(data=time_cond, dtype=data_type)
    try:
        logger.info("Dataset tensors are created on current device for single-GPU efficiency.")
    except Exception:
        pass
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

    train_batch_size = 12  # 可根据显存情况调整以提升吞吐
    val_batch_size = 10
    test_batch_size = 10
    logger.info(f"Batch size of train, val, and test: {train_batch_size}, {val_batch_size}, {test_batch_size}")
    # 默认微批大小：在不改变 DataLoader 的情况下，减半批次以降低峰值显存
    try:
        Par["micro_batch"] = int(Par.get("micro_batch", 0))  # 保守：默认不使用微批
    except Exception:
        Par["micro_batch"] = 0
    # DataLoader：默认关闭多进程与共享内存，避免 worker 段错误；可通过 Par 覆盖
    _cpu_ct = os.cpu_count() or 2
    _default_workers = max(1, _cpu_ct - 1)
    num_workers_train = int(Par.get("num_workers_train", 0))
    num_workers_eval = int(Par.get("num_workers_eval", 0))
    use_shared_mem = bool(Par.get("use_shared_memory", False))
    logger.info(
        f"DataLoader config -> train_workers: {num_workers_train}, eval_workers: {num_workers_eval}, use_shared_memory: {use_shared_mem}"
    )
    train_loader = paddle.io.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers_train,
        use_shared_memory=use_shared_mem,
    )
    val_loader = paddle.io.DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers_eval,
        use_shared_memory=use_shared_mem,
    )
    test_loader = paddle.io.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers_eval,
        use_shared_memory=use_shared_mem,
    )

    model = Unet2D_with_FNO(
        dim=8,
        Par=Par,
        dim_mults=(1, 2, 4),
        channels=Par["channels"],
    ).astype("float32")

    # 输出层初始化：使归一化后的初始预测接近0，便于稳定起步
    try:
        with paddle.no_grad():
            if hasattr(model, "final_conv"):
                if getattr(model.final_conv, "weight", None) is not None:
                    w = model.final_conv.weight
                    model.final_conv.weight.set_value(paddle.zeros_like(w))
                if getattr(model.final_conv, "bias", None) is not None:
                    b = model.final_conv.bias
                    model.final_conv.bias.set_value(paddle.zeros_like(b))
    except Exception:
        pass

    # 单卡模式：不使用 DataParallel 封装，直接在当前设备训练
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        logger.info(f"Using device(s): {visible if visible else 'auto-detected'}")
    except Exception:
        pass

    # 初始化损失函数，传入机翼掩码与当前进程的 logger，避免多进程下的全局引用问题
    criterion = CustomLoss(Par, mask=Par.get('mask'), logger=logger)
    # 初始化优化器 - 优先在并行包装之后创建，以正确获取并行参数
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=1e-06,
        weight_decay=1e-06,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=0.05),
    )
    # 初始化学习率调度器
    scheduler = combined_scheduler(
        optimizer,
        Par["num_epochs"] * len(train_loader),
        int(0.1 * Par["num_epochs"]) * len(train_loader),
    )

    # Training loop
    num_epochs = Par["num_epochs"]
    # 频率控制（可通过 Par 覆盖）；默认不进行额外验证与绘图以降低开销
    VAL_EVERY = int(Par.get("val_every", 0))
    PLOT_EVERY = int(Par.get("plot_every", 0))
    # 默认限制验证批次数，避免过长的验证过程
    MAX_VAL_BATCHES = int(Par.get("val_batches", 10))
    VAL_MAX_STEPS = Par.get("val_max_steps", None)
    TRAIN_MAX_BATCHES = Par.get("train_batches", None)
    amp_enable = bool(Par.get("amp_enable", False))  # 保守：默认禁用 AMP
    best_val_loss = float("inf")
    best_model_id = 0
    # 记录每个 epoch 的损失以绘制曲线
    train_losses = []
    val_losses = []  # 滚动验证损失
    val_1step_losses = []  # 单步验证损失
    
    # 在主文件夹下创建models和images子文件夹
    models_dir = os.path.join(main_output_dir, "models")
    images_dir = os.path.join(main_output_dir, "images")
    if is_main:
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        # 初始化损失CSV并写入表头
        loss_csv_path = os.path.join(main_output_dir, "loss_curve.csv")
        try:
            with open(loss_csv_path, "w", encoding="utf8") as f:
                f.write("epoch,train_loss,val_rollout_loss,val_1step_loss\n")
        except Exception as e:
            logger.warning(f"Failed to init loss CSV: {e}")
    else:
        loss_csv_path = os.path.join(main_output_dir, "loss_curve.csv")
    t0 = time.time()
    # AMP 与 cuDNN 性能设置：提升吞吐
    scaler = paddle.amp.GradScaler()
    try:
        paddle.set_flags({"FLAGS_cudnn_deterministic": False, "FLAGS_benchmark": True})
        logger.info("Enabled cuDNN benchmark and disabled deterministic.")
    except Exception:
        pass
    phys_warmup_epochs = max(30, int(0.2 * num_epochs))
    warmup_steps = phys_warmup_epochs * len(train_loader)
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        begin_time = time.time()
        model.train()
        train_loss = 0.0
        try:
            logger.info(f"[R{rank_id}] Start epoch {epoch}/{num_epochs} on device {paddle.device.get_device()}")
        except Exception:
            pass
        tb = 0
        for x_idx, t_idx, y_idx in tqdm(
            train_loader,
            desc=f"[R{rank_id}] Epoch {epoch}/{num_epochs}",
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not is_main,
        ): 
            # 将批次索引移动到与数据张量一致的设备，避免在 DataLoader worker 中初始化 CUDA
            x_idx_dev = paddle.to_tensor(x_idx.numpy(), dtype='int64', place=traj_train_tensor.place)
            t_idx_dev = paddle.to_tensor(t_idx.numpy(), dtype='int64', place=time_cond_tensor.place)
            y_idx_dev = paddle.to_tensor(y_idx.numpy(), dtype='int64', place=traj_train_tensor.place)
            x = traj_train_tensor[0, x_idx_dev]
            t = time_cond_tensor[t_idx_dev]
            y_true = traj_train_tensor[0, y_idx_dev]
            optimizer.clear_gradients(set_to_zero=False)

            warmup_factor = min(1.0, global_step / max(1, warmup_steps))
            try:
                criterion.set_warmup_factor(warmup_factor)
            except Exception:
                pass

            try:
                x = x.astype('float32')
                t = t.astype('float32')
                y_true = y_true.astype('float32')
            except Exception:
                pass

            # 微批梯度累积：降低峰值显存占用，缓解 Einsum/Matmul 反传 OOM
            micro_bs = int(Par.get("micro_batch", 0))
            B = int(x.shape[0])
            use_micro = micro_bs > 0 and micro_bs < B
            batch_loss_value = 0.0
            if use_micro:
                accum_steps = int(math.ceil(B / micro_bs))
                w_loss, w_denom = 0.0, 0
                try:
                    for s in range(0, B, micro_bs):
                        e = min(B, s + micro_bs)
                        x_i = x[s:e]
                        t_i = t[s:e]
                        y_true_i = y_true[s:e]
                        with paddle.amp.auto_cast(enable=amp_enable):
                            y_pred_i = model(x_i, t_i)
                        try:
                            pred_ok = bool(paddle.isfinite(y_pred_i).all().numpy())
                        except Exception:
                            pred_ok = True
                        if not pred_ok:
                            with paddle.amp.auto_cast(enable=False):
                                y_pred_i = model(x_i.astype('float32'), t_i.astype('float32'))
                            try:
                                pred_ok = bool(paddle.isfinite(y_pred_i).all().numpy())
                            except Exception:
                                pred_ok = True
                            if not pred_ok:
                                y_pred_i = paddle.nan_to_num(y_pred_i, nan=0.0, posinf=0.0, neginf=0.0)
                                try:
                                    criterion.set_warmup_factor(0.0)
                                except Exception:
                                    pass
                        loss_i = criterion(y_pred_i.astype('float32'), y_true_i.astype('float32'))
                        if not np.isfinite(float(loss_i.item())):
                            try:
                                criterion.set_warmup_factor(0.0)
                            except Exception:
                                pass
                            loss_i = criterion(y_pred_i.astype('float32'), y_true_i.astype('float32'))
                            if not np.isfinite(float(loss_i.item())):
                                optimizer.clear_gradients(set_to_zero=True)
                                continue
                        # 归一化到累积步，避免梯度放大
                        scaled_chunk = scaler.scale(loss_i / accum_steps)
                        scaled_chunk.backward()
                        w_loss += float(loss_i.item()) * (e - s)
                        w_denom += (e - s)
                except MemoryError:
                    # 显存不足时跳过此批，避免训练中断
                    logger.warning(f"[R{rank_id}] OOM during micro-batch backward; skipping batch and lowering warmup factor")
                    try:
                        criterion.set_warmup_factor(0.0)
                    except Exception:
                        pass
                    optimizer.clear_gradients(set_to_zero=True)
                    tb += 1
                    global_step += 1
                    scheduler.step()
                    continue
                batch_loss_value = (w_loss / max(1, w_denom)) if w_denom > 0 else 0.0
                scaler.step(optimizer)
                scaler.update()
            else:
                with paddle.amp.auto_cast(enable=amp_enable):
                    y_pred = model(x, t)
                try:
                    pred_is_finite = bool(paddle.isfinite(y_pred).all().numpy())
                except Exception:
                    pred_is_finite = True
                if not pred_is_finite:
                    logger.warning("[R%d] Non-finite prediction detected; retrying forward in FP32 (AMP disabled)" % rank_id)
                    with paddle.amp.auto_cast(enable=False):
                        y_pred = model(x.astype('float32'), t.astype('float32'))
                    try:
                        pred_is_finite = bool(paddle.isfinite(y_pred).all().numpy())
                    except Exception:
                        pred_is_finite = True
                    if not pred_is_finite:
                        y_pred = paddle.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                        try:
                            criterion.set_warmup_factor(0.0)
                        except Exception:
                            pass
                loss = criterion(y_pred.astype('float32'), y_true.astype('float32'))
                if not np.isfinite(float(loss.item())):
                    logger.warning("Non-finite loss detected; applying MSE-only fallback for this batch")
                    try:
                        criterion.set_warmup_factor(0.0)
                    except Exception:
                        pass
                    fallback_loss = criterion(y_pred.astype('float32'), y_true.astype('float32'))
                    if not np.isfinite(float(fallback_loss.item())):
                        optimizer.clear_gradients(set_to_zero=True)
                        continue
                    loss = fallback_loss
                try:
                    scaled = scaler.scale(loss)
                    scaled.backward()
                except MemoryError:
                    logger.warning(f"[R{rank_id}] OOM during backward; skipping batch and lowering warmup factor")
                    try:
                        criterion.set_warmup_factor(0.0)
                    except Exception:
                        pass
                    optimizer.clear_gradients(set_to_zero=True)
                    tb += 1
                    global_step += 1
                    scheduler.step()
                    continue
                try:
                    grads = []
                    for p in model.parameters():
                        if p.grad is not None:
                            grads.append(paddle.flatten(p.grad))
                    if grads and epoch <= 5:
                        total_norm = float(paddle.linalg.norm(paddle.concat(grads), p=2).numpy())
                except Exception:
                    pass
                scaler.step(optimizer)
                scaler.update()
                batch_loss_value = float(loss.item())
            # 累积训练损失（按批平均）
            train_loss += batch_loss_value
            scheduler.step()
            global_step += 1
            tb += 1
            if TRAIN_MAX_BATCHES is not None and tb >= int(TRAIN_MAX_BATCHES):
                break

        train_loss /= len(train_loader)
        if is_main:
            train_losses.append(train_loss)

        # Validation（仅主进程执行，避免重复计算与文件写入冲突）
        model.eval()
        do_val = VAL_EVERY > 0 and (epoch % VAL_EVERY == 0 or epoch == num_epochs)
        if is_main and do_val:
            val_rollout_loss = 0.0
            val_onestep_loss = 0.0
            with paddle.no_grad():
                vb = 0
                # 将验证时间索引移动到设备
                t_idx_val_dev = paddle.to_tensor(t_idx_val.numpy(), dtype='int64', place=time_cond_tensor.place)
                for x_idx, y_idx in val_loader:
                    if vb >= MAX_VAL_BATCHES:
                        break
                    x_idx_dev = paddle.to_tensor(x_idx.numpy(), dtype='int64', place=traj_val_tensor.place)
                    y_idx_dev = paddle.to_tensor(y_idx.numpy(), dtype='int64', place=traj_val_tensor.place)
                    x = traj_val_tensor[0, x_idx_dev]
                    # 滚动验证使用完整的时间条件向量 (长度 lf)
                    t_roll = time_cond_tensor[t_idx_val_dev]
                    y_true = traj_val_tensor[0, y_idx_dev]
                    NT = Par["lb"] + (VAL_MAX_STEPS if VAL_MAX_STEPS else y_true.shape[1])
                    y_pred = rollout(model, x, t_roll, NT, Par, val_batch_size)
                    loss_roll = criterion(y_pred.astype('float32'), y_true.astype('float32'))
                    val_rollout_loss += loss_roll.item()
                    vb += 1

                    # 单步验证
                    t0_scalar = time_cond_tensor[t_idx_val_dev[0]]
                    t_batch = t0_scalar.reshape([1]).tile(repeat_times=[x.shape[0]])
                    y_true_1 = traj_val_tensor[0, y_idx_dev[:, 0]]
                    y_pred_1 = model(x.astype('float32'), t_batch.astype('float32'))
                    loss_1 = criterion(y_pred_1.astype('float32'), y_true_1.astype('float32'))
                    val_onestep_loss += loss_1.item()
                denom = min(len(val_loader), MAX_VAL_BATCHES) if len(val_loader) > 0 else 0
                val_rollout_avg = val_rollout_loss / denom if denom > 0 else float('nan')
                val_1step_avg = val_onestep_loss / denom if denom > 0 else float('nan')
                val_losses.append(val_rollout_avg)
                val_1step_losses.append(val_1step_avg)
                if PLOT_EVERY > 0 and ((epoch == 1) or (epoch == num_epochs) or (epoch % PLOT_EVERY == 0)):
                    try:
                        make_plot(
                            y_true.detach().cpu().numpy(),
                            y_pred.detach().cpu().numpy(),
                            epoch,
                            images_dir,
                            train_losses=train_losses,
                            val_rollout_losses=val_losses,
                            val_1step_losses=val_1step_losses,
                            mask2d=mask[0, 0],
                        )
                    except Exception as e:
                        logger.warning(f"Plotting failed at epoch {epoch}: {e}")
            val_loss = val_rollout_avg if 'val_rollout_avg' in locals() else float('nan')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_id = epoch
                paddle.save(obj=model.state_dict(), path=os.path.join(models_dir, "best_model.pdparams"))
            elapsed_time = time.time() - begin_time
            logger.info(
                f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Rollout Loss: {val_loss:.4e}, Val 1step Loss: {val_1step_avg:.4e}, "
                f"Best model: {best_model_id}, Learning rate: {scheduler.get_lr():.4e}, "
                f"Epoch time: {elapsed_time:.2f}"
            )
            # 每个epoch结束后追加一行到CSV
            try:
                with open(loss_csv_path, "a", encoding="utf8") as f:
                    f.write(f"{epoch},{train_loss:.6e},{val_loss:.6e},{val_1step_avg:.6e}\n")
            except Exception as e:
                logger.warning(f"Failed to append loss CSV: {e}")
        # 单卡模式：无需分布式同步

    logger.info("Training finished.")
    logger.info(f"Training Time: {time.time() - t0:.1f}s")
    
    # 测试集评估（仅主进程，默认关闭）
    if is_main and bool(Par.get("run_test", False)):
        model.eval()
        test_loss = 0.06
        with paddle.no_grad():
            # 将测试时间索引移动到设备
            t_idx_test_dev = paddle.to_tensor(t_idx_test.numpy(), dtype='int64', place=time_cond_tensor.place)
            for x_idx, y_idx in test_loader:
                x_idx_dev = paddle.to_tensor(x_idx.numpy(), dtype='int64', place=traj_test_tensor.place)
                y_idx_dev = paddle.to_tensor(y_idx.numpy(), dtype='int64', place=traj_test_tensor.place)
                x = traj_test_tensor[0, x_idx_dev]
                t = time_cond_tensor[t_idx_test_dev]
                y_true = traj_test_tensor[0, y_idx_dev]
                NT = Par["lb"] + y_true.shape[1]
                y_pred = rollout(model, x, t, NT, Par, val_batch_size)
                loss = criterion(y_pred.astype('float32'), y_true.astype('float32'))
                test_loss += loss.item()
        test_loss /= len(test_loader)
        logger.info(f"Test Loss: {test_loss:.4e}")


if __name__ == "__main__":
    # 纯单卡入口：直接运行
    main_worker(rank=None)
