import json
import logging
import math
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paddle
import scipy.stats as stats
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from ppcfd.models.ppdiffusion.matcho import Unet2D


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
    return logger


def make_plot(TRUE, PRED, epoch, images_dir="images", train_losses=None, val_losses=None):
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
        Tt = paddle.to_tensor(data=Ta, dtype=data_type)
        Pt = paddle.to_tensor(data=Pa, dtype=data_type)
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


    fig, axes = plt.subplots(6, N, figsize=(30, 14))
    axes = np.array(axes)
    cbar_ax = fig.add_axes([0.92, 0.26, 0.02, 0.62])

    for i in range(N):
        vmin_i, vmax_i = get_vmin_vmax(true[i], pred1[i])
        if not np.isfinite(vmin_i) or not np.isfinite(vmax_i) or vmin_i == vmax_i:
            base = 1e-6 if (not np.isfinite(vmin_i) or not np.isfinite(vmax_i)) else max(1e-6, abs(vmin_i) * 1e-3 + 1e-6)
            vmin_i = (0.0 if not np.isfinite(vmin_i) else vmin_i) - base
            vmax_i = (0.0 if not np.isfinite(vmax_i) else vmax_i) + base
        im = axes[0, i].imshow(true[i], vmin=vmin_i, vmax=vmax_i, cmap=CMAP)
        axes[0, i].set_title(f"Time: {int(time_ls[i])}s (Ux)", fontsize=14)
        axes[0, i].axis("off")

        im = axes[1, i].imshow(pred1[i], vmin=vmin_i, vmax=vmax_i, cmap=CMAP)
        mse_val1 = rel_l2(true[i], pred1[i])
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

        # Uy 通道：图像
        vmin_u, vmax_u = get_vmin_vmax(true_uy[i], pred_uy[i])
        if not np.isfinite(vmin_u) or not np.isfinite(vmax_u) or vmin_u == vmax_u:
            base = 1e-6 if (not np.isfinite(vmin_u) or not np.isfinite(vmax_u)) else max(1e-6, abs(vmin_u) * 1e-3 + 1e-6)
            vmin_u = (0.0 if not np.isfinite(vmin_u) else vmin_u) - base
            vmax_u = (0.0 if not np.isfinite(vmax_u) else vmax_u) + base
        im2 = axes[3, i].imshow(true_uy[i], vmin=vmin_u, vmax=vmax_u, cmap=CMAP)
        axes[3, i].set_title(f"Time: {int(time_ls[i])}s (Uy)", fontsize=14)
        axes[3, i].axis("off")

        im2 = axes[4, i].imshow(pred_uy[i], vmin=vmin_u, vmax=vmax_u, cmap=CMAP)
        mse_val_u = rel_l2(true_uy[i], pred_uy[i])
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
    if train_losses is not None and val_losses is not None:
        try:
            loss_ax = fig.add_axes([0.08, 0.04, 0.8, 0.14])
            x = np.arange(1, len(train_losses) + 1)
            val_x = np.arange(1, len(val_losses) + 1)
            train_arr = np.asarray(train_losses, dtype=float)
            val_arr = np.asarray(val_losses, dtype=float)
            train_arr = np.where(np.isfinite(train_arr), train_arr, np.nan)
            val_arr = np.where(np.isfinite(val_arr), val_arr, np.nan)
            loss_ax.plot(x, train_arr, label="Train loss", color="tab:blue", linewidth=2)
            loss_ax.plot(val_x, val_arr, label="Val loss", color="tab:orange", linestyle="--", marker="o", markersize=3, alpha=0.9, linewidth=2, zorder=3)
            loss_ax.set_yscale("log")
            loss_ax.set_xlabel("Epoch")
            loss_ax.set_ylabel("Loss (log)")
            loss_ax.set_title(f"Loss curves through epoch {epoch + 1}")
            loss_ax.grid(True, which="both", ls=":", alpha=0.3)
            loss_ax.legend(loc="upper right")
        except Exception:
            pass

    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(os.path.join(images_dir, f"{epoch + 1}.png"))
    plt.close(fig)


class CustomLoss(paddle.nn.Layer):
    def __init__(self, Par):
        super(CustomLoss, self).__init__()
        self.Par = Par

    def forward(self, y_pred, y_true):
        y_true = (y_true - self.Par["out_shift"]) / (self.Par["out_scale"])
        y_pred = (y_pred - self.Par["out_shift"]) / (self.Par["out_scale"])
        numer = paddle.linalg.norm(x=y_true - y_pred, p=2)
        denom = paddle.linalg.norm(x=y_true, p=2)
        loss = numer / paddle.clip(denom, min=1e-8)
        return loss


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
    
    # 在主文件夹下创建子文件夹
    save_dir = os.path.join(main_output_dir, f"seed_{seed_value}")
    logger = init_all(seed_value, name=save_dir, dtype=data_type)

    # AMP 动态损失缩放：仅缩放损失（auto_cast 关闭），训练早期若溢出会自动降低缩放系数。
    # AMP 已禁用：采用纯 FP32 训练，无需 GradScaler

    begin_time = time.time()
    ux_data_dir = "/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251022_143447/UX_nan_filtered.npy"
    uy_data_dir = "/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251022_143447/UY_nan_filtered.npy"
    mask_dir = "/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251022_143447/mask.npy"
    logger.info(f"Data paths: ux={ux_data_dir}, uy={uy_data_dir}")

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
    Par["lf"] = 3 #lookforward 单次预测的时间窗口长度
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
    logger.info(f"Input shift of trai_train: {Par['inp_shift']}")
    logger.info(f"Input scale of trai_train: {Par['inp_scale']}")
    logger.info(f"Output shift of trai_train: {Par['out_shift']}")
    logger.info(f"Output scale of trai_train: {Par['out_scale']}")
    logger.info(f"Time shift: {Par['t_shift']}")
    logger.info(f"Time scale: {Par['t_scale']}")
    logger.info(f"Time cond: {Par['time_cond']}\n")

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

    train_batch_size = 20  # 100
    val_batch_size = 20  # 100
    test_batch_size = 20  # 100
    logger.info(f"Batch size of train, val, and test: {train_batch_size}, {val_batch_size}, {test_batch_size}")
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = paddle.io.DataLoader(dataset=val_dataset, batch_size=val_batch_size)
    test_loader = paddle.io.DataLoader(dataset=test_dataset, batch_size=test_batch_size)
    model = Unet2D(
        dim=16,
        Par=Par,
        dim_mults=(1, 2, 4, 8),
        channels=Par["channels"],
    ).astype("float32")

    # 初始化损失函数
    criterion = CustomLoss(Par)
    # 初始化优化器
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=5 * 1e-05,
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
                logger.warning("Non-finite loss detected; skipping this batch")
                optimizer.clear_gradients(set_to_zero=True)
                continue

            loss.backward()
            try:
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(paddle.flatten(p.grad))
                if grads and epoch <= 5:
                    total_norm = float(paddle.linalg.norm(paddle.concat(grads), p=2).numpy())
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
