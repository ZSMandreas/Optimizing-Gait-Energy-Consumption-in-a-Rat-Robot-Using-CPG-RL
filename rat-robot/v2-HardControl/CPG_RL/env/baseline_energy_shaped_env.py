from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from cpg_shape_action_env import RatCPGEnvShapeAction


class RatCpgEnvEnergySubstep50ShapeV2(RatCPGEnvShapeAction):
    """Route-A shape env v2 with stronger energy/saturation defaults."""

    def __init__(self, *args, xvel_boost_multiplier: float = 1.0, **kwargs):
        kwargs.setdefault("joint_energy_weight", 0.30)
        kwargs.setdefault("neg_power_alpha", 2.0)
        kwargs.setdefault("sat_weight", 5.0)
        kwargs.setdefault("sat_threshold", 0.75)
        kwargs.setdefault("smooth_weight", 0.10)
        super().__init__(*args, **kwargs)
        self.xvel_boost_multiplier = float(xvel_boost_multiplier)
        if self.xvel_boost_multiplier != 1.0:
            print(f"[W2 env init] xvel_boost_multiplier = {self.xvel_boost_multiplier}")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        if self.xvel_boost_multiplier != 1.0:
            r_xvel_orig = None
            if "reward/r_xvel" in info:
                r_xvel_orig = float(info["reward/r_xvel"])
            elif "r_xvel" in info:
                r_xvel_orig = float(info["r_xvel"])
            if r_xvel_orig is not None:
                extra = r_xvel_orig * (self.xvel_boost_multiplier - 1.0)
                reward = float(reward) + extra
                info["reward/r_xvel_boost_extra"] = float(extra)
                info["reward/r_xvel_total"] = float(r_xvel_orig + extra)
                info["r_xvel"] = float(r_xvel_orig + extra)
                if hasattr(self, "_ep_comp_sums") and isinstance(self._ep_comp_sums, dict):
                    self._ep_comp_sums["r_xvel"] = float(self._ep_comp_sums.get("r_xvel", 0.0) + extra)
                if (terminated or truncated) and isinstance(info.get("episode_components"), dict):
                    info["episode_components"]["ep_r_xvel"] = float(self._ep_comp_sums.get("r_xvel", 0.0))
                    ep_total = info["episode_components"].get("ep_total_reward", None)
                    if ep_total is not None:
                        info["episode_components"]["ep_total_reward"] = float(ep_total + extra)
            else:
                info["reward/r_xvel_boost_extra"] = 0.0
        return obs, reward, terminated, truncated, info
