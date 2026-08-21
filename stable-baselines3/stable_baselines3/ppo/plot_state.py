import numpy as np
import matplotlib.pyplot as plt
import math

def plot_measured(pos_ms, vel_ms):
    
    pos_ms = np.stack(pos_ms)
    
    # flattened_pos_ms = pos_ms[:,0,:]
    
    vel_ms = np.stack(vel_ms)
    # print(f"pos_ms shape: {pos_ms.shape}")
    # flattened_vel_ms = vel_ms[:,0,:]


    fig, axes = plt.subplots(7, 3, figsize=(10, 15))
    fig.suptitle('State (Position)', fontsize=14)

    # labels = [
    #     'Base Position X', 'Base Position Y', 'Base Position Z',
    #     'Base Orientation Roll', 'Base Orientation Pitch', 'Base Orientation Yaw',
    #     'FL Position X', 'FL Position Y', 'FL Position Z',
    #     'FR Position X', 'FR Position Y', 'FR Position Z',
    #     'RL Position X', 'RL Position Y', 'RL Position Z',
    #     'RR Position X', 'RR Position Y', 'RR Position Z'
    # ]

    # # Plot positions (base pos, orientation, legs)
    # for i in range():
    #     row, col = divmod(i, 3)
    #     # axes[row, col].plot(pos_ds[:, i], label='Desired Position')
    #     axes[row, col].plot(pos_ms[:, i], label='Measured Position')
    #     axes[row, col].set_title(labels[i])
    #     axes[row, col].set_xlabel('Time Step')
    #     axes[row, col].set_ylabel('Position')
    #     axes[row, col].legend()
    
    labels = [
        'FL_thigh', 'FL_calf', 'FR_thigh',
        'FR_calf', 'RL_thigh', 'RL_calf',
        'RR_thigh','RR_calf',
        'Tail', 'Neck', 'Head',
        'Spine', 'FL_foot', 'FR_foot',
        'RL_foot', 'RL_foot',
        'Position X', 'Position Y', 'Position Z'
    ]

    # Plot positions (base pos, orientation, legs)
    for i in range(19):
        row, col = divmod(i, 3)
        # axes[row, col].plot(pos_ds[:, i], label='Desired Position')
        axes[row, col].plot(pos_ms[:, i], label='Measured Position')
        axes[row, col].set_title(labels[i])
        axes[row, col].set_xlabel('Time Step')
        axes[row, col].set_ylabel('Position')
        axes[row, col].legend()

    # Add a figure for velocities
    fig_vel, axes_vel = plt.subplots(3, 3, figsize=(15, 18))
    fig_vel.suptitle('State (Velocities)', fontsize=14)

    vel_labels = [
        'Base Velocity X', 'Base Velocity Y', 'Base Velocity Z',
        'Acc X','Acc Y','Acc Z',
        'Angular Velocity Roll', 'Angular Velocity Pitch', 'Angular Velocity Yaw',
       
    ]

    # Plot velocities (base vel, angular vel, legs)
    for i in range(9):
        row, col = divmod(i, 3)
        # axes_vel[row, col].plot(vel_ds[:, i], label='Desired Velocity')
        axes_vel[row, col].plot(vel_ms[:, i], label='Measured Velocity')
        axes_vel[row, col].set_title(vel_labels[i])
        axes_vel[row, col].set_xlabel('Time Step')
        axes_vel[row, col].set_ylabel('Velocity')
        axes_vel[row, col].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

class VarianceTracker:
    def __init__(self, obs_dim=32):
        self.obs_dim = obs_dim
        self.history = []  # List of 32-dim variance vectors
        var_buffer = []

    def update(self, buffer):
        """
        每次更新时调用，传入新的观测 buffer(list of np.array(shape=(32,)))
        """

        obs_array = np.stack(buffer, axis=0)  # shape: (50, 32)
        # print(obs_array.shape)
        var = np.var(obs_array, axis=0)       # shape: (32,)
        self.history.append(var)
        # print(f"self.history{self.history}")

    def savebuffer(self):
        var_array = np.stack(self.history, axis=0)
        return var_array

    def plot(self, title="Variance over time (each dimension)", figsize=(16, 12), show=True, smooth_window=9):
        """
        每个维度一个 subplot,展示其方差随时间的变化
        """

        if len(self.history) == 0:
            print("⚠️ No history to plot.")
            return

        var_array = np.stack(self.history, axis=0)  # shape: (num_steps, obs_dim)
        num_steps, obs_dim = var_array.shape

        # 设置子图的行列数（比如 8 行 × 4 列）
        cols = 4
        rows = math.ceil(obs_dim / cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()  # 变成 1D list，方便访问
        
        def smooth_curve(data, window_size=smooth_window):
            if len(data) < window_size:
                return data
            return np.convolve(data, np.ones(window_size)/window_size, mode='same')

        for i in range(obs_dim):
            ax = axes[i]
            smoothed = smooth_curve(var_array[:, i], smooth_window)
            ax.plot(smoothed)
            ax.set_title(f"Dimension {i}", fontsize=10)
            ax.set_xlabel("Step")
            ax.set_ylabel("Var")
            ax.grid(True)

        # 删除多余的 subplot（如果 obs_dim 不是整除）
        for j in range(obs_dim, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if show:
            plt.show()
        




from sklearn.metrics import r2_score

def compare_trajectories(file1, file2, label1="training", label2="controller"):
    """
    对比两个头部坐标轨迹 .npy 文件，绘制轨迹图，计算平均欧氏距离和 R² 准确率。

    Args:
        file1 (str): 第一个 .npy 文件路径
        file2 (str): 第二个 .npy 文件路径
        label1 (str): 第一个轨迹的标签
        label2 (str): 第二个轨迹的标签
    """
    # 加载数据
    head_1 = np.load(file1)  # shape: (T, 2) or (T, 3)
    real_start1 = head_1[0]
    real_end1 = head_1[-1]
    # print(real_end)
    head_2 = np.load(file2)
    real_start2 = head_2[0]
    real_end2 = head_2[-1]
   

    # # 截断为相同长度
    # min_len = min(len(head_1), len(head_2))
    # head_1 = head_1[:min_len]
    # head_2 = head_2[:min_len]


    # # 计算欧氏距离差异
    # diff = np.linalg.norm(head_1 - head_2, axis=1)
    # avg_diff = np.mean(diff)

    # # 计算 R²（分别对 X 和 Y）
    # r2_x = r2_score(head_1[:, 0], head_2[:, 0])
    # r2_y = r2_score(head_1[:, 1], head_2[:, 1])
    # r2_avg = (r2_x + r2_y) / 2

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(head_1[:, 0], head_1[:, 1], label=label1, color='blue')
    plt.plot([real_start1[0], real_end1[0]], [real_start1[1], real_end1[1]], 'b--', label="Start-End Line")
    # plt.scatter(*real_start1, color='green', label="Start_no")
    # plt.scatter(*real_end1, color='red', label="End_no")
    plt.plot(head_2[:, 0], head_2[:, 1], label=label2, color='red')
    plt.plot([real_start2[0], real_end2[0]], [real_start2[1], real_end2[1]], 'r--', label="Start-End Line")
    # plt.scatter(*real_start2, color='green', label="Start_with")
    # plt.scatter(*real_end2, color='red', label="End_with")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Body Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    real_distance_no_spine = np.linalg.norm(real_end1 - real_start1)
    # real_distance_with_spine = np.linalg.norm(real_end2 - real_start2)
    print(f"real distance without spine: {real_distance_no_spine}")
    # print(f"real distance with spine: {real_distance_with_spine}")
    # # 打印指标
    # print(f"平均轨迹差异（欧氏距离）: {avg_diff:.4f}")
    # print(f"R² 准确率 (X): {r2_x:.4f},(Y): {r2_y:.4f}，平均: {r2_avg:.4f}")

# 示例调用（放在 main 中）
if __name__ == '__main__':
    compare_trajectories(
        "/home/geriatronics/ratmujoco_ws/real_trajectory_no_spine.npy",
        # "/home/simiao/semester arbeit/real_trajectory_with_spine.npy"
        "/home/geriatronics/ratmujoco_ws/path_xy.npy"
    )

