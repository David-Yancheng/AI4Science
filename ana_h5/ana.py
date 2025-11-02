import os
from datetime import datetime

import numpy as np
import h5py

TARGET_SAMPLE_ELEMS = 250_000  # 目标采样元素数量，避免一次性读取超大数据集


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024 or u == "TB":
            return f"{size:.2f} {u}" if u != "B" else f"{int(size)} {u}"
        size /= 1024


def dataset_stats(ds: h5py.Dataset):
    """对数据集做采样统计（min/mean/max）。非数值类型返回 None。失败时返回错误字符串。"""
    try:
        dtype = ds.dtype
        # i: int, u: uint, f: float, b: bool
        if dtype.kind not in "iufb":
            return None
        shape = ds.shape
        # 标量数据集
        if shape == ():
            arr = np.asarray(ds[()])
        else:
            total = int(np.prod(shape))
            if total <= TARGET_SAMPLE_ELEMS:
                # 小数据集直接读取
                arr = np.asarray(ds[()])
            else:
                # 大数据集按各维步长采样，目标总样本量约为 TARGET_SAMPLE_ELEMS
                ndim = len(shape)
                # 计算每维步长，使得采样量约为 total / factor ≈ TARGET_SAMPLE_ELEMS
                factor = max(1, int(np.ceil(total / TARGET_SAMPLE_ELEMS)))
                base = max(1, int(np.ceil(factor ** (1.0 / ndim))))
                slices = tuple(slice(0, s, base) for s in shape)
                arr = np.asarray(ds[slices])
        # 转为 float64 便于统计
        arr = arr.astype(np.float64, copy=False)
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        mean = float(np.mean(arr))
        return mn, mean, mx, arr.size, arr.shape
    except Exception as e:
        return f"统计失败: {type(e).__name__}: {e}"


def build_report(file_path: str) -> str:
    lines = []
    lines.append(f"文件: {file_path}")
    try:
        size = os.path.getsize(file_path)
        lines.append(f"大小: {human_size(size)}")
    except Exception:
        pass

    with h5py.File(file_path, "r") as f:
        # 根属性
        if len(f.attrs) > 0:
            keys = list(f.attrs.keys())
            lines.append(f"根属性键: {keys}")

        ds_infos = []
        grp_count = 0

        def visitor(name, obj):
            nonlocal grp_count
            if isinstance(obj, h5py.Dataset):
                info = {
                    "path": "/" + name,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "size": obj.size,
                    "nbytes": obj.size * obj.dtype.itemsize,
                    "chunks": obj.chunks,
                    "compression": obj.compression,
                    "compression_opts": obj.compression_opts,
                    "attrs": list(obj.attrs.keys()),
                    "stats": dataset_stats(obj),
                }
                ds_infos.append(info)
            elif isinstance(obj, h5py.Group):
                grp_count += 1

        f.visititems(visitor)
        lines.append(f"分组数: {grp_count}，数据集数: {len(ds_infos)}")

        # 按数据集元素数降序展示前 30 个
        ds_infos.sort(key=lambda d: d["size"], reverse=True)
        max_show = min(len(ds_infos), 30)
        for i in range(max_show):
            d = ds_infos[i]
            lines.append(f"- 数据集 {i + 1}/{len(ds_infos)}: {d['path']}")
            lines.append(
                f"  形状: {d['shape']}，类型: {d['dtype']}，元素数: {d['size']}，估计大小: {human_size(d['nbytes'])}"
            )
            lines.append(
                f"  块: {d['chunks']}，压缩: {d['compression']}，压缩参数: {d['compression_opts']}"
            )
            if d["attrs"]:
                lines.append(f"  属性键: {d['attrs']}")
            stats = d["stats"]
            if isinstance(stats, tuple):
                mn, mean, mx, sample_elems, sample_shape = stats
                lines.append(
                    f"  采样统计(样本 {sample_elems}，形状{sample_shape}): min={mn:.6g}, mean={mean:.6g}, max={mx:.6g}"
                )
            elif stats is None:
                lines.append("  非数值数据，跳过统计")
            else:
                lines.append(f"  统计信息: {stats}")

        if len(ds_infos) > max_show:
            lines.append(f"... 其余 {len(ds_infos) - max_show} 个数据集省略")

    return "\n".join(lines)


def main():
    grid_path = r"c:\Users\Harry\Desktop\project\ana_h5\airfoilLES_grid.h5"
    frame_path = r"c:\Users\Harry\Desktop\project\ana_h5\airfoilLES_t00010.h5"

    reports = []
    for p in [grid_path, frame_path]:
        if not os.path.exists(p):
            reports.append(f"文件不存在: {p}")
        else:
            reports.append(build_report(p))

    header = "=" * 80
    full = (
        "\n\n"
        + header
        + "\n分析时间: "
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + "\n"
        + header
        + "\n\n"
        + "\n\n".join(reports)
        + "\n"
    )

    print(full)

    # 把结果追加写入 README.md
    try:
        readme_path = r"c:\Users\Harry\Desktop\project\ana_h5\README.md"
        with open(readme_path, "a", encoding="utf-8") as fp:
            fp.write("\n\n## 自动分析结果\n")
            fp.write(full)
        print(f"\n结果已追加到: {readme_path}")
    except Exception as e:
        print(f"写入 README.md 失败: {e}")


if __name__ == "__main__":
    main()