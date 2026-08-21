import numpy as np
import matplotlib.pyplot as plt
import os

# 加载状态数据
data = np.load("/home/simiao/semester arbeit/5_state_data.npz")
states_t = data["states_t"]  # shape: (T, N)
fs = 100  # 采样频率（Hz）

# 输出文件夹
output_dir = "fft_bar_plots"
os.makedirs(output_dir, exist_ok=True)

# 只分析前12维
for i in range(32):
    signal = states_t[:, i]
    N = len(signal)

    # 傅里叶变换
    freq = np.fft.rfftfreq(N, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(signal))

    # 绘制离散频谱柱状图
    plt.figure(figsize=(8, 4))
    plt.bar(freq, fft_mag, width=fs/N * 0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(f"FFT of Dimension {i}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fft_bar_dim_{i}.png"))
    plt.close()

print(f"保存完毕，所有图像存于文件夹：{output_dir}/")
