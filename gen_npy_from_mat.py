import scipy.io
import numpy as np

# 读取 CWRU 105.mat 文件，提取振动信号并保存为 .npy
mat_path = r"凯斯西储大学数据/12k Drive End Bearing Fault Data/105.mat"
out_path = "test_signal.npy"

mat = scipy.io.loadmat(mat_path)
# CWRU 数据常见信号键名有 'DE_time', 'FE_time', 'BA_time'，优先用 DE_time

# 自动查找包含 'DE' 和 'time' 的键（区分大小写）
signal = None
for key in mat:
    if 'DE' in key and 'time' in key:
        signal = mat[key].squeeze()
        print(f"使用信号键: {key}")
        break
if signal is None:
    raise ValueError(f"未找到包含 'DE' 和 'time' 的信号键，mat文件包含: {list(mat.keys())}")

np.save(out_path, signal)
print(f"已保存 {out_path}，长度: {len(signal)}")
