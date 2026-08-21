from __future__ import annotations

import math
from typing import Dict

import numpy as np

from w2_energy_shaped_env import RatCpgEnvEnergySubstep50ShapeV2


class RatCpgEnvEnergySubstep50ShapeV3(RatCpgEnvEnergySubstep50ShapeV2):
    """W3 Ours v3.1: replace W2 energy term and optionally boost xvel penalty."""

    DEFAULT_WEIGHTS = {
        "alpha_W_pos": 0.30,
        "alpha_W_neg": 0.60,
        "alpha_E_damp": 0.50,
        "alpha_E_fric": 0.50,
        "alpha_E_norm": 1.00,
    }

    def __init__(
        self,
        *args,
        ours_weights: dict | None = None,
        xvel_boost_multiplier: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ours_weights = {**self.DEFAULT_WEIGHTS, **(ours_weights or {})}
        self.xvel_boost_multiplier = float(xvel_boost_multiplier)
        print(f"[v3.1 env init] xvel_boost_multiplier = {self.xvel_boost_multiplier}")
        print(f"[v3.1 env init] Ours reward weights = {self.ours_weights}")

    def _compute_step_energy_components(self) -> Dict[str, float]:
        tau8 = np.asarray(self.data.actuator_force[self._actuator_idx_leg], dtype=np.float64)  # [N*m]
        qvel8 = np.asarray(self.data.qvel[self._dof_idx_leg], dtype=np.float64)  # [rad/s]
        dt_env = float(self.dt) * float(self.n_substeps)  # [s]
        b_damp = 0.005  # [N*m*s/rad]

        p_inst = tau8 * qvel8  # [W]
        w_pos = float(np.sum(np.maximum(p_inst, 0.0))) * dt_env  # [J]
        w_neg = float(np.sum(np.maximum(-p_inst, 0.0))) * dt_env  # [J]
        e_damp = float(b_damp * np.sum(qvel8 ** 2)) * dt_env  # [J]
        e_fric = float(getattr(self, "_contact_E_friction_step_acc", 0.0))  # [J]
        e_norm = float(getattr(self, "_contact_E_normal_step_acc", 0.0))  # [J]
        return {"W_pos": w_pos, "W_neg": w_neg, "E_damp": e_damp, "E_fric": e_fric, "E_norm": e_norm}

    def _compute_ours_energy_reward_term(self) -> tuple[float, Dict[str, float]]:
        e = self._compute_step_energy_components()
        w = self.ours_weights
        r_energy = -(
            w["alpha_W_pos"] * e["W_pos"]
            + w["alpha_W_neg"] * e["W_neg"]
            + w["alpha_E_damp"] * e["E_damp"]
            + w["alpha_E_fric"] * e["E_fric"]
            + w["alpha_E_norm"] * e["E_norm"]
        )
        info = {
            "energy/W_pos_J": e["W_pos"],
            "energy/W_neg_J": e["W_neg"],
            "energy/E_damp_J": e["E_damp"],
            "energy/E_fric_J": e["E_fric"],
            "energy/E_norm_J": e["E_norm"],
            "reward/r_energy": float(r_energy),
        }
        return float(r_energy), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        old_r_joint_energy = float(info.get("r_joint_energy", 0.0))
        old_weighted_term = float(self.joint_energy_weight) * old_r_joint_energy

        ours_energy_reward, extra_info = self._compute_ours_energy_reward_term()
        reward = float(reward) - old_weighted_term + ours_energy_reward

        # Keep callback-compatible component stream aligned to Ours term.
        info["r_joint_energy"] = float(ours_energy_reward / max(float(self.joint_energy_weight), 1e-9))
        info.update(extra_info)

        # Adjust episode-level component summary to reflect replacement.
        if hasattr(self, "_ep_comp_sums") and isinstance(self._ep_comp_sums, dict):
            self._ep_comp_sums["r_joint_energy"] += float(info["r_joint_energy"] - old_r_joint_energy)
        if (terminated or truncated) and isinstance(info.get("episode_components"), dict):
            info["episode_components"]["ep_r_joint_energy"] = float(self._ep_comp_sums.get("r_joint_energy", 0.0))
            ep_total = info["episode_components"].get("ep_total_reward", None)
            if ep_total is not None:
                # Replace last-step contribution approximation for end-of-episode report.
                info["episode_components"]["ep_total_reward"] = float(ep_total - old_weighted_term + ours_energy_reward)

        # Optional post-hoc xvel boost (v3.1). Keep default 1.0 identical to v3 behavior.
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
                info["reward/r_xvel_total"] = float(info.get("r_xvel", 0.0))
        return obs, reward, terminated, truncated, info

# Clearer alias for CDER Ours v3.1
OursCderEnvV31 = RatCpgEnvEnergySubstep50ShapeV3
