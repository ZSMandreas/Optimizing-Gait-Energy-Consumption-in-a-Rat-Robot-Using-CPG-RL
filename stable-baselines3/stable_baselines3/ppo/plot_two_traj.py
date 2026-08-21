import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/simiao/rat_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/src/speed_scan_logs"
FRES = [0.513, 1.25]  # 你要画的两条
TAG = lambda fre: f"fre_{fre:.3f}".replace(".", "p")

paths = {}
for fre in FRES:
    d = os.path.join(ROOT, TAG(fre))
    f = os.path.join(d, "path_xyz.npy")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing: {f}")
    paths[fre] = np.load(f)  # [T,3]

# 1) top-down x-y
plt.figure()
for fre, p in paths.items():
    plt.plot(p[:,0], p[:,1], label=f"fre={fre:.3f}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory (top-down x-y)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "traj_xy_compare.png"), dpi=200)
plt.close()

# 2) y over time
plt.figure()
for fre, p in paths.items():
    plt.plot(np.arange(len(p)), p[:,1], label=f"fre={fre:.3f}")
plt.xlabel("step index")
plt.ylabel("y (m)")
plt.title("Forward displacement (y) over time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "traj_y_time_compare.png"), dpi=200)
plt.close()

print("Saved:",
      os.path.join(ROOT, "traj_xy_compare.png"),
      os.path.join(ROOT, "traj_y_time_compare.png"))
