import matplotlib.pyplot as plt
import numpy as np
import math


def compare_variance_plot(var_array_n50, var_array_n5, figsize=(20, 16), smooth_window=5, title="n=50 vs n=5 "):
    num_steps, obs_dim = var_array_n50.shape
    cols = 4
    rows = math.ceil(obs_dim / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(obs_dim):
        ax = axes[i]
        smooth_n50 = var_array_n50
        smooth_n5 = var_array_n5
        if i == 0:
            ax.plot(smooth_n50, label="n=50", color="blue")
            ax.plot(smooth_n5, label="n=5", color="orange")
            ax.legend(fontsize=8)
        else:
            ax.plot(smooth_n50, color="blue")
            ax.plot(smooth_n5, color="orange")

        ax.set_title(f"Dim {i}", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Var")
        ax.grid(True)
        ax.legend(fontsize=8)

    for j in range(obs_dim, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=10)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    var_array_n50 = np.load("var_array_n50.npy")
    var_array_n5 = np.load("var_array_n5.npy")
    compare_variance_plot(var_array_n50, var_array_n5)