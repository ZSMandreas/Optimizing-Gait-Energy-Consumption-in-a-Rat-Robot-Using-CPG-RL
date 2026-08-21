import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfftfreq
from scipy.fft import rfft, irfft
import os
os.chdir("/home/simiao/rat_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models")

foot_condition_log = np.load("/home/simiao/rat_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/src/speed_scan_logs/fre_0p513/touch_substep.npy")  # shape: [T, 4]
binary_contact = (foot_condition_log > 0).astype(int)

plt.figure(figsize=(12, 3))
plt.imshow(binary_contact.T, cmap='Greys', aspect='auto', interpolation='nearest')
plt.yticks([0, 1, 2, 3], ["FL", "FR", "HL", "HR"])
plt.xlabel("Time Step")
plt.title("Foot Contact Pattern (Black = Contact)")
plt.colorbar(label="Contact")
plt.show()

# # 加载用户最新上传的 foot_condition.npy 文件并进行主频分析和绘图
# binary_contact = np.load("foot_condition_with_spine.npy").T
# leg_names = ["FL", "FR", "RL", "RR"]
# N = binary_contact.shape[1]
# t = np.arange(N)
# sample_rate = 100.0  # 假设采样率为 100Hz
# xf = rfftfreq(N, d=1/sample_rate)

# # 主频提取与频谱图绘制
# main_freq_info = {}
# fig, axs = plt.subplots(4, 1, figsize=(12, 6), sharex=True)

# for i, leg in enumerate(leg_names):
#     y = binary_contact[i]
#     yf = rfft(y)
#     amplitudes = np.abs(yf)

#     # 主频提取（跳过0Hz）
#     peak_index = np.argmax(amplitudes[1:]) + 1
#     main_freq = xf[peak_index]
#     amplitude = amplitudes[peak_index]
#     phase = np.angle(yf[peak_index])

#     main_freq_info[leg] = {
#         "Main Frequency (Hz)": main_freq,
#         "Amplitude": amplitude,
#         "Phase (rad)": phase
#     }

#     # 绘制频谱图（柱状图 + 主频标红）
#     axs[i].bar(xf, amplitudes, width=0.1, color="blue", alpha=0.6)
#     axs[i].bar(xf[peak_index], amplitudes[peak_index], width=0.1, color="red")
#     axs[i].set_xlim(0, 20)
#     axs[i].set_ylabel(leg)
#     axs[i].set_title(f"{leg} | Main Freq: {main_freq:.2f} Hz, Phase: {phase:.2f} rad")

# axs[-1].set_xlabel("Frequency (Hz)")
# plt.suptitle("Fourier Spectrum and Main Frequency Analysis (Binary Contact Signal)")
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

# # 输出主频分析结果
# import pandas as pd
# main_freq_df = pd.DataFrame(main_freq_info).T
# # import ace_tools as tools; tools.display_dataframe_to_user(name="Main Frequency Analysis (foot_condition.npy)", dataframe=main_freq_df)
