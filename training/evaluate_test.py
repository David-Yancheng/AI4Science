import argparse
import json
import os
import time
import numpy as np
import paddle
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppcfd.models.ppdiffusion.matcho import Unet2D


def setup_device():
    try:
        gpu_available = bool(paddle.is_compiled_with_cuda())
        try:
            gpu_count = paddle.device.cuda.device_count()
        except Exception:
            gpu_count = 0
        device = "gpu" if gpu_available and gpu_count > 0 else "cpu"
        paddle.device.set_device(device)
    except Exception:
        device = "cpu"
    return device


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


def preprocess(traj, Par):
    nt = traj.shape[1]
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
    # sliding_window_view for targets
    from numpy.lib.stride_tricks import sliding_window_view
    y_idx = sliding_window_view(y_idx[Par["lb"] :], window_shape=effective_LF)
    return (
        paddle.to_tensor(data=x_idx, dtype="int64"),
        paddle.to_tensor(data=t_idx, dtype="int64"),
        paddle.to_tensor(data=y_idx, dtype="int64"),
    )


def rollout(model, x, t, NT, Par, batch_size):
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
            with paddle.no_grad():
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


def rel_l2_np(true, pred):
    numer = np.linalg.norm(true - pred)
    denom = np.linalg.norm(true)
    return float(numer / (denom + 1e-8))


def mae_np(true, pred):
    return float(np.mean(np.abs(true - pred)))


def mse_np(true, pred):
    diff = true - pred
    return float(np.mean(diff * diff))


def make_plot(TRUE, PRED, epoch, images_dir="images_test", summary=None, note=None):
    # Ensure plotting runs even without CJK fonts; fall back cleanly
    try:
        chosen_font = setup_chinese_font()
    except NameError:
        # Fallback: keep minus sign visible; CJK glyphs may be missing
        try:
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
        chosen_font = None
        print("[warn] setup_chinese_font() not found. Using fallback settings; consider merging latest evaluate_test.py or adding the font setup function.")
    if chosen_font is None:
        # One-time notice (does not affect image generation)
        print("[info] No CJK font detected. Labels remain in English to avoid tofu boxes on Linux.")
    sample_id = 0
    T = TRUE.shape[1]
    skip_t = max(1, T // 8)
    idx_all = np.arange(T)[::skip_t]
    # 展示帧数量（最多3列）
    N = min(idx_all.shape[0], 3)
    idx = idx_all[:N]
    # 指标平均使用的帧数量（最多10帧，等间隔采样）
    N_avg = min(idx_all.shape[0], 10)
    idx_avg = idx_all[:N_avg]
    true = np.nan_to_num(TRUE[sample_id, idx, 0], nan=0.0, posinf=0.0, neginf=0.0)
    pred1 = np.nan_to_num(PRED[sample_id, idx, 0], nan=0.0, posinf=0.0, neginf=0.0)
    true_uy = np.nan_to_num(TRUE[sample_id, idx, 1], nan=0.0, posinf=0.0, neginf=0.0)
    pred_uy = np.nan_to_num(PRED[sample_id, idx, 1], nan=0.0, posinf=0.0, neginf=0.0)
    CMAP = "turbo"

    def get_vmin_vmax(a, b):
        vals_a = a[np.isfinite(a)]
        vals_b = b[np.isfinite(b)]
        stack = np.concatenate([vals_a.reshape(-1), vals_b.reshape(-1)])
        if stack.size == 0:
            vmin = float(np.nanmin(np.stack([a, b])))
            vmax = float(np.nanmax(np.stack([a, b])))
            return vmin, vmax
        vmin = np.percentile(stack, 1)
        vmax = np.percentile(stack, 99)
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

    fig, axes = plt.subplots(6, N, figsize=(30, 15))
    axes = np.array(axes)
    cbar_ax = fig.add_axes([0.92, 0.26, 0.02, 0.62])
    for i in range(N):
        vmin_i, vmax_i = get_vmin_vmax(true[i], pred1[i])
        im = axes[0, i].imshow(true[i], vmin=vmin_i, vmax=vmax_i, cmap=CMAP)
        axes[0, i].set_title("Ux True")
        axes[0, i].axis("off")
        axes[1, i].imshow(pred1[i], vmin=vmin_i, vmax=vmax_i, cmap=CMAP)
        axes[1, i].set_title(f"Ux relL2: {rel_l2_np(true[i], pred1[i]):.2e}")
        axes[1, i].axis("off")
        # Ux spectrum
        image = true[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2).flatten()
        Abins_true, _, _ = stats.binned_statistic(knrm, fourier_amplitudes.flatten(), statistic="mean", bins=np.arange(0.5, min(nx, ny)//2 + 1, 1.0))
        kbins = np.arange(0.5, min(nx, ny)//2 + 1, 1.0)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins_true *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_true = np.maximum(Abins_true, 1e-12)
        image = pred1[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        Abins_pred, _, _ = stats.binned_statistic(knrm, fourier_amplitudes.flatten(), statistic="mean", bins=kbins)
        Abins_pred *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_pred = np.maximum(Abins_pred, 1e-12)
        axes[2, i].loglog(kvals, Abins_true, 'b-', label="True Ux")
        axes[2, i].loglog(kvals, Abins_pred, 'r--', label="Pred Ux")
        k_ref = np.linspace(1, np.max(kbins), 100)
        energy_ref = k_ref ** (-5 / 3)
        energy_ref *= max(max(Abins_true), max(Abins_pred)) / max(energy_ref)
        axes[2, i].loglog(k_ref, energy_ref, "k:", label="k^-5/3", alpha=0.7)
        axes[2, i].legend()
        # Uy images and spectrum
        vmin_u, vmax_u = get_vmin_vmax(true_uy[i], pred_uy[i])
        axes[3, i].imshow(true_uy[i], vmin=vmin_u, vmax=vmax_u, cmap=CMAP)
        axes[3, i].set_title("Uy True")
        axes[3, i].axis("off")
        axes[4, i].imshow(pred_uy[i], vmin=vmin_u, vmax=vmax_u, cmap=CMAP)
        axes[4, i].set_title(f"Uy relL2: {rel_l2_np(true_uy[i], pred_uy[i]):.2e}")
        axes[4, i].axis("off")
        image = true_uy[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2).flatten()
        Abins_true, _, _ = stats.binned_statistic(knrm, fourier_amplitudes.flatten(), statistic="mean", bins=np.arange(0.5, min(nx, ny)//2 + 1, 1.0))
        kbins = np.arange(0.5, min(nx, ny)//2 + 1, 1.0)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins_true *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_true = np.maximum(Abins_true, 1e-12)
        image = pred_uy[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        Abins_pred, _, _ = stats.binned_statistic(knrm, fourier_amplitudes.flatten(), statistic="mean", bins=kbins)
        Abins_pred *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_pred = np.maximum(Abins_pred, 1e-12)
        axes[5, i].loglog(kvals, Abins_true, label="True Uy")
        axes[5, i].loglog(kvals, Abins_pred, label="Pred Uy")
        k_ref = np.linspace(1, np.max(kbins), 100)
        energy_ref = k_ref ** (-5 / 3)
        energy_ref *= max(max(Abins_true), max(Abins_pred)) / max(energy_ref)
        axes[5, i].loglog(k_ref, energy_ref, "k--", label="k^-5/3")
        axes[5, i].legend()
    fig.colorbar(im, cax=cbar_ax)

    # =====================
    # 指标汇总与备注叠加到图像
    # =====================
    def avg_metrics_over_frames(true_seq, pred_seq):
        rels, maes, mses = [], [], []
        for i in range(true_seq.shape[0]):
            rels.append(rel_l2_np(true_seq[i], pred_seq[i]))
            maes.append(mae_np(true_seq[i], pred_seq[i]))
            mses.append(mse_np(true_seq[i], pred_seq[i]))
        return float(np.mean(rels)), float(np.mean(maes)), float(np.mean(mses))

    # 使用N_avg帧进行平均指标计算
    true_avg_ux = np.nan_to_num(TRUE[sample_id, idx_avg, 0], nan=0.0, posinf=0.0, neginf=0.0)
    pred_avg_ux = np.nan_to_num(PRED[sample_id, idx_avg, 0], nan=0.0, posinf=0.0, neginf=0.0)
    true_avg_uy = np.nan_to_num(TRUE[sample_id, idx_avg, 1], nan=0.0, posinf=0.0, neginf=0.0)
    pred_avg_uy = np.nan_to_num(PRED[sample_id, idx_avg, 1], nan=0.0, posinf=0.0, neginf=0.0)
    ux_rel, ux_mae, ux_mse = avg_metrics_over_frames(true_avg_ux, pred_avg_ux)
    uy_rel, uy_mae, uy_mse = avg_metrics_over_frames(true_avg_uy, pred_avg_uy)

    global_rel = global_mae = global_mse = None
    if isinstance(summary, dict):
        try:
            global_rel = float(summary.get("relL2_overall_mean")) if summary.get("relL2_overall_mean") is not None else None
            global_mae = float(summary.get("MAE_overall_mean")) if summary.get("MAE_overall_mean") is not None else None
            global_mse = float(summary.get("MSE_overall_mean")) if summary.get("MSE_overall_mean") is not None else None
        except Exception:
            pass

    steps_note = (
        "Notes:\n"
        "- Metrics computed inside mask; color range is symmetric with 1%–99% percentile auto-scaling.\n"
        "- Display frames: evenly sample N frames from T for visualization; averages use up to 10 evenly spaced frames.\n"
        "- Spectrum: 2D FFT, radial binning, multiplied by annulus area; overlay k^-5/3 reference line."
    )
    if isinstance(note, str) and len(note.strip()) > 0:
        steps_note += "\n" + note.strip()

    metrics_text = (
        f"Averaged (over {N_avg} evenly sampled frames):\n"
        f"Ux: relL2={ux_rel:.3e}, MAE={ux_mae:.3e}, MSE={ux_mse:.3e}\n"
        f"Uy: relL2={uy_rel:.3e}, MAE={uy_mae:.3e}, MSE={uy_mse:.3e}"
    )
    if (global_rel is not None) or (global_mae is not None) or (global_mse is not None):
        metrics_text += (
            "\n" +
            f"Global test mean: relL2={global_rel if global_rel is not None else 'NA'}, "
            f"MAE={global_mae if global_mae is not None else 'NA'}, "
            f"MSE={global_mse if global_mse is not None else 'NA'}"
        )

    info_ax = fig.add_axes([0.05, 0.02, 0.86, 0.22])
    info_ax.axis("off")
    info_ax.text(0.0, 0.95, metrics_text, fontsize=10, va="top")
    info_ax.text(0.0, 0.45, steps_note, fontsize=9, va="top")
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, f"epoch_{epoch}.png"))
    plt.close(fig)


def evaluate(ux_path, uy_path, mask_path, par_json_path, model_path, output_dir, batch_size=20):
    os.makedirs(output_dir, exist_ok=True)
    device = setup_device()
    print(f"Device: {device}")
    # Load data
    traj_ux = np.load(ux_path)
    traj_uy = np.load(uy_path)
    traj_ux = np.expand_dims(traj_ux, axis=0)
    traj_uy = np.expand_dims(traj_uy, axis=0)
    if traj_ux.shape != traj_uy.shape:
        raise ValueError(f"Ux and Uy shapes differ: {traj_ux.shape} vs {traj_uy.shape}")
    traj = np.stack([traj_ux, traj_uy], axis=2)  # [N, T, C, nx, ny]
    # 将掩膜应用到数据上（用于指标计算与可视化），并在 Par 中保留为 Paddle 张量供模型使用
    mask_np = np.load(mask_path)
    # 规范到 [1, 1, nx, ny] 便于与数据广播
    if mask_np.ndim == 2:
        mask_np = mask_np[None, None, ...]
    elif mask_np.ndim == 3:
        # 可能为 [1, nx, ny]
        mask_np = mask_np[:, None, ...]
    elif mask_np.ndim == 4:
        # 可能已为 [1, 1, nx, ny]
        pass
    else:
        raise ValueError(f"Unexpected mask shape: {mask_np.shape}")
    mask_np = mask_np.astype(np.float32)
    traj = traj * mask_np

    # Split train/val/test as in training script (80/10/10)
    nt_all = traj.shape[1]
    train_end = int(nt_all * 0.8)
    val_end = int(nt_all * 0.9)
    train_end = max(1, min(train_end, nt_all - 2))
    val_end = max(train_end + 1, min(val_end, nt_all - 1))
    traj_test = traj[:, val_end:]

    # Load Par
    with open(par_json_path, "r", encoding="utf8") as f:
        Par = json.load(f)
    # Ensure numeric types
    Par["nx"] = int(traj_test.shape[-2])
    Par["ny"] = int(traj_test.shape[-1])
    Par["nf"] = int(2)
    Par["lb"] = int(Par.get("lb", 10))
    Par["lf"] = int(Par.get("lf", 2))
    Par["LF"] = int(Par.get("LF", 10))
    time_cond = np.asarray(Par.get("time_cond", np.linspace(0, 1, Par["lf"])), dtype=np.float32)

    # 在 Par 中设置用于模型前向的掩膜张量，形状匹配 [nf, nx, ny]
    # 将 2D 掩膜复制到通道维度
    mask_2d = mask_np.squeeze()  # [nx, ny]
    if mask_2d.ndim != 2:
        mask_2d = mask_2d.reshape(Par["nx"], Par["ny"])  # 兜底，确保为 [nx, ny]
    mask_for_model = np.repeat(mask_2d[None, :, :], repeats=Par["nf"], axis=0).astype(np.float32)  # [nf, nx, ny]
    Par["mask"] = paddle.to_tensor(mask_for_model, dtype="float32")

    # Build tensors
    traj_test_tensor = paddle.to_tensor(data=traj_test, dtype="float32")
    time_cond_tensor = paddle.to_tensor(data=time_cond, dtype="float32")

    # Indices
    x_idx_test, t_idx_test, y_idx_test = preprocess(traj_test, Par)

    # Model
    model = Unet2D(
        dim=16,
        Par=Par,
        dim_mults=(1, 2, 4, 8),
        channels=Par["nf"] * Par["lb"],
    ).astype("float32")
    state_dict = paddle.load(model_path)
    model.set_state_dict(state_dict)
    model.eval()

    # Evaluation
    criterion = CustomLoss(Par)
    metrics_csv = os.path.join(output_dir, "metrics_test.csv")
    summary_json = os.path.join(output_dir, "summary_test.json")
    images_dir = os.path.join(output_dir, "images_test")
    preds_path = os.path.join(output_dir, "pred_test.npy")
    with open(metrics_csv, "w", encoding="utf8") as f:
        f.write("timestep,relL2_Ux,relL2_Uy,MAE_Ux,MAE_Uy,relL2_mean,MAE_mean,MSE_Ux,MSE_Uy,MSE_mean\n")

    test_loss = 0.0
    all_pred_samples = []
    begin = time.time()
    with paddle.no_grad():
        # We iterate over all index rows sequentially without DataLoader to keep script simple
        total_windows = x_idx_test.shape[0]
        # process in chunks of batch_size
        for start in range(0, total_windows, batch_size):
            end = min(start + batch_size, total_windows)
            x = traj_test_tensor[0, x_idx_test[start:end]]
            t = time_cond_tensor[t_idx_test]
            y_true = traj_test_tensor[0, y_idx_test[start:end]]
            NT = Par["lb"] + y_true.shape[1]
            y_pred = rollout(model, x, t, NT, Par, batch_size)
            loss = criterion(y_pred.astype("float32"), y_true.astype("float32")).item()
            test_loss += loss
            # metrics per timestep
            y_true_np = y_true.numpy()
            y_pred_np = y_pred.numpy()
            # collect first sample for plotting
            if len(all_pred_samples) < 1:
                all_pred_samples.append((y_true_np, y_pred_np))
            # aggregate across batch for metrics
            B, Tm, C, nx, ny = y_true_np.shape
            for tstep in range(Tm):
                true_t = y_true_np[:, tstep]
                pred_t = y_pred_np[:, tstep]
                # channel metrics averaged over batch
                rels = []
                maes = []
                mses = []
                for ch in range(C):
                    rel = 0.0
                    mae_v = 0.0
                    mse_v = 0.0
                    for b in range(B):
                        rel += rel_l2_np(true_t[b, ch], pred_t[b, ch])
                        mae_v += mae_np(true_t[b, ch], pred_t[b, ch])
                        mse_v += mse_np(true_t[b, ch], pred_t[b, ch])
                    rel /= B
                    mae_v /= B
                    mse_v /= B
                    rels.append(rel)
                    maes.append(mae_v)
                    mses.append(mse_v)
                rel_mean = float(np.mean(rels))
                mae_mean = float(np.mean(maes))
                mse_mean = float(np.mean(mses))
                with open(metrics_csv, "a", encoding="utf8") as f:
                    f.write(f"{tstep},{rels[0]:.6e},{rels[1]:.6e},{maes[0]:.6e},{maes[1]:.6e},{rel_mean:.6e},{mae_mean:.6e},{mses[0]:.6e},{mses[1]:.6e},{mse_mean:.6e}\n")
            # Save predictions incrementally (optional concat)
            # We store only last chunk to keep file moderate; adjust if needed
            last_chunk = y_pred_np

    test_loss /= max(1, (total_windows + batch_size - 1) // batch_size)
    # summary
    # read back CSV to compute averages quickly
    data = np.genfromtxt(metrics_csv, delimiter=",", skip_header=1)
    if data.ndim == 1 and data.size > 0:
        data = np.expand_dims(data, 0)
    summary = {
        "test_loss_relL2": float(test_loss),
        "relL2_Ux_mean": float(np.mean(data[:, 1])) if data.size else None,
        "relL2_Uy_mean": float(np.mean(data[:, 2])) if data.size else None,
        "MAE_Ux_mean": float(np.mean(data[:, 3])) if data.size else None,
        "MAE_Uy_mean": float(np.mean(data[:, 4])) if data.size else None,
        "relL2_overall_mean": float(np.mean(data[:, 5])) if data.size else None,
        "MAE_overall_mean": float(np.mean(data[:, 6])) if data.size else None,
        "MSE_Ux_mean": float(np.mean(data[:, 7])) if data.size else None,
        "MSE_Uy_mean": float(np.mean(data[:, 8])) if data.size else None,
        "MSE_overall_mean": float(np.mean(data[:, 9])) if data.size else None,
        "elapsed_sec": float(time.time() - begin),
    }
    with open(summary_json, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    # plots
    if all_pred_samples:
        y_true_sample, y_pred_sample = all_pred_samples[0]
        # 将全局测试均值传入绘图函数，用于在图中叠加指标说明
        make_plot(y_true_sample, y_pred_sample, epoch=0, images_dir=images_dir, summary=summary)
    # save last prediction block
    if 'last_chunk' in locals():
        np.save(preds_path, last_chunk)

    print(f"Test evaluation done. Report dir: {output_dir}")


def main():
    # Set your defaults here; command-line args will override if provided
    DEFAULT_MODEL_PTH = r"/data/zhouziyue_benkesheng/AI4SC_program/training/train_results_20251023_191353/models/best_model.pdparams"
    DEFAULT_PAR_JSON = r"/data/zhouziyue_benkesheng/AI4SC_program/training/train_results_20251023_191353/Par.json"
    DEFAULT_UX_PATH = r"/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251022_143447/UX_nan_filtered.npy"
    DEFAULT_UY_PATH = r"/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251022_143447/UY_nan_filtered.npy"
    DEFAULT_MASK_PATH = r"/data/zhouziyue_benkesheng/AI4SC_program/data_pre/results_20251022_143447/mask.npy"
    DEFAULT_OUTPUT_DIR = r"/data/zhouziyue_benkesheng/AI4SC_program/training/train_results_20251025_112833/test_report"
    DEFAULT_BATCH_SIZE = 20

    parser = argparse.ArgumentParser(description="Evaluate best model on test flow fields with fine-grained metrics")
    parser.add_argument("--model_pth", default=DEFAULT_MODEL_PTH, help="Path to best_model.pdparams")
    parser.add_argument("--par_json", default=DEFAULT_PAR_JSON, help="Path to Par.json from training")
    parser.add_argument("--ux_path", default=DEFAULT_UX_PATH, help="Path to Ux .npy file")
    parser.add_argument("--uy_path", default=DEFAULT_UY_PATH, help="Path to Uy .npy file")
    parser.add_argument("--mask_path", default=DEFAULT_MASK_PATH, help="Path to binary mask .npy file")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write metrics and plots")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for rollout evaluation")
    args = parser.parse_args()

    evaluate(
        ux_path=args.ux_path,
        uy_path=args.uy_path,
        mask_path=args.mask_path,
        par_json_path=args.par_json,
        model_path=args.model_pth,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
def setup_chinese_font():
    """Configure a font that supports Chinese to avoid square boxes.
    Prefer system fonts; fallback to loading a local font file if present.
    Returns the chosen font name or None if not found.
    """
    from matplotlib import font_manager
    import os

    # 1) 尝试系统可用字体
    candidates = [
        "Microsoft YaHei",  # Windows 常见
        "SimHei",           # 黑体
        "SimSun",           # 宋体
        "Noto Sans CJK SC", # 谷歌思源/Noto
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return name
    except Exception:
        pass

    # 2) 回退：扫描项目内的常见中文字体文件并注册
    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        here,
        os.path.join(here, "assets", "fonts"),
        os.path.join(here, "fonts"),
        os.path.abspath(os.path.join(here, "..", "assets", "fonts")),
        os.path.abspath(os.path.join(here, "..", "fonts")),
        os.getcwd(),
    ]
    font_files = [
        "NotoSansCJKsc-Regular.otf",
        "NotoSansCJK-Regular.ttc",
        "SourceHanSansSC-Regular.otf",
        "SimHei.ttf",
        "MicrosoftYaHei.ttf",
        "MSYH.TTC",
    ]
    for d in search_dirs:
        for fname in font_files:
            fp = os.path.join(d, fname)
            if os.path.isfile(fp):
                try:
                    font_manager.fontManager.addfont(fp)
                    prop = font_manager.FontProperties(fname=fp)
                    name = prop.get_name()
                    plt.rcParams["font.sans-serif"] = [name]
                    plt.rcParams["axes.unicode_minus"] = False
                    print(f"[info] 已加载本地中文字体文件: {fp} -> {name}")
                    return name
                except Exception as e:
                    print(f"[warn] 加载本地字体失败: {fp}: {e}")

    # 3) 最小设置，保证负号正常显示（可能仍显示方块）
    plt.rcParams["axes.unicode_minus"] = False
    return None