"""
RatCPGEnvShapeAction — substep50 env with shared (a, b) in RL action.

Based on RatCPGEnv (base, no IDER).
Keeps diagonal coupling utilities (ab_phase_rel_obs + w_sync).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from gymnasium import spaces

from rat_cpg_env_energy_substep50 import RatCPGEnv


class RatCPGEnvShapeAction(RatCPGEnv):
    ACTION_RANGES_EXT = {
        "f": (0.3, 2.5),
        "mu": (0.2, 1.0),
        "a": (0.010, 0.029),
        "b": (0.001, 0.008),
    }

    def __init__(self, *args, w_sync: float = 0.0, ab_phase_rel_obs: bool = True, **kwargs):
        kwargs.setdefault("use_energy_tank", False)
        super().__init__(*args, **kwargs)

        self.ACTION_RANGES = dict(self.ACTION_RANGES_EXT)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        self.w_sync = float(w_sync)
        self.ab_phase_rel_obs = bool(ab_phase_rel_obs)

        self._cur_a = float(np.mean(self.ACTION_RANGES["a"]))
        self._cur_b = float(np.mean(self.ACTION_RANGES["b"]))

        base_dim = 35
        obs_dim = base_dim + 2 + (4 if self.ab_phase_rel_obs else 0)
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _map_action_10(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        a = np.clip(action.astype(np.float64), -1.0, 1.0).reshape(10)
        f_lo, f_hi = self.ACTION_RANGES["f"]
        m_lo, m_hi = self.ACTION_RANGES["mu"]
        a_lo, a_hi = self.ACTION_RANGES["a"]
        b_lo, b_hi = self.ACTION_RANGES["b"]
        f4 = f_lo + (a[0:4] + 1.0) * 0.5 * (f_hi - f_lo)
        mu4 = m_lo + (a[4:8] + 1.0) * 0.5 * (m_hi - m_lo)
        a_s = a_lo + (a[8] + 1.0) * 0.5 * (a_hi - a_lo)
        b_s = b_lo + (a[9] + 1.0) * 0.5 * (b_hi - b_lo)
        return f4, mu4, float(a_s), float(b_s)

    def _phase_rel_features(self) -> List[float]:
        p = [float(self.foot_path.get_leg_phase(i)) for i in range(4)]
        phi_A = math.atan2(math.sin(p[0]) + math.sin(p[3]), math.cos(p[0]) + math.cos(p[3]))
        phi_B = math.atan2(math.sin(p[1]) + math.sin(p[2]), math.cos(p[1]) + math.cos(p[2]))
        dphi = phi_A - phi_B
        return [math.sin(dphi), math.cos(dphi), math.sin(phi_A), math.cos(phi_A)]

    def _q_cmd_8(self) -> np.ndarray:
        out = np.zeros(8, dtype=np.float64)
        for leg_i, (hip_act, knee_act) in enumerate(self.actuator_ids):
            out[2 * leg_i + 0] = float(self.data.ctrl[int(hip_act)]) if hip_act >= 0 else 0.0
            out[2 * leg_i + 1] = float(self.data.ctrl[int(knee_act)]) if knee_act >= 0 else 0.0
        return out

    @staticmethod
    def _r_sync_from_qref(qref: np.ndarray) -> float:
        qv = np.asarray(qref, dtype=np.float64).ravel()
        if qv.size < 8:
            return 0.0
        d1 = qv[0:2] - qv[6:8]
        d2 = qv[2:4] - qv[4:6]
        return -0.5 * (float(np.mean(d1 * d1)) + float(np.mean(d2 * d2)))

    def _append_shape_to_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, np.array([self._cur_a, self._cur_b], dtype=np.float32)], axis=0)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._cur_a = float(np.mean(self.ACTION_RANGES["a"]))
        self._cur_b = float(np.mean(self.ACTION_RANGES["b"]))
        self.foot_path.set_shape_uniform(self._cur_a, self._cur_b)
        if self.ab_phase_rel_obs:
            obs = np.concatenate([obs, np.array(self._phase_rel_features(), dtype=np.float32)], axis=0)
        obs = self._append_shape_to_obs(obs)
        return obs, info

    def step(self, action: np.ndarray):
        if action.shape != (10,):
            raise ValueError(f"action must be (10,), got {action.shape}")

        if self.control_step_counter % self.action_update_interval == 0:
            if self.fixed_cpg:
                f4_tgt = np.full(4, 0.5, dtype=np.float64)
                mu4_tgt = np.full(4, 1.0, dtype=np.float64)
                a_tgt = 0.020
                b_tgt = 0.004
            else:
                f4_tgt, mu4_tgt, a_tgt, b_tgt = self._map_action_10(action)

            alpha = float(self.smoothing_alpha)
            self._cur_f4 = self._smooth_vec(self._cur_f4, f4_tgt)
            self._cur_mu4 = self._smooth_vec(self._cur_mu4, mu4_tgt)
            self._cur_a = (1.0 - alpha) * self._cur_a + alpha * float(a_tgt)
            self._cur_b = (1.0 - alpha) * self._cur_b + alpha * float(b_tgt)
            self.foot_path.set_controls(self._cur_f4, self._cur_mu4)
            self.foot_path.set_shape_uniform(self._cur_a, self._cur_b)
            self._last_applied_action = np.array(action, dtype=np.float64)

        # Base env expects action dim 8 and updates controls internally.
        # Temporarily disable internal action update to avoid overriding shape controls.
        old_interval = self.action_update_interval
        try:
            self.action_update_interval = 10**9
            obs, reward, terminated, truncated, info = super().step(np.zeros(8, dtype=np.float32))
        finally:
            self.action_update_interval = old_interval

        qref = self._q_cmd_8()
        r_sync = self._r_sync_from_qref(qref)
        info["r_sync"] = float(r_sync)
        if self.w_sync > 0.0:
            sync_term = self.w_sync * float(r_sync)
            reward = float(reward) + sync_term
            self._ep_total_reward += sync_term

        if self.ab_phase_rel_obs:
            obs = np.concatenate([obs, np.array(self._phase_rel_features(), dtype=np.float32)], axis=0)
        obs = self._append_shape_to_obs(obs)

        info["action_a"] = float(self._cur_a)
        info["action_b"] = float(self._cur_b)
        if (terminated or truncated) and isinstance(info.get("episode_components"), dict):
            info["episode_components"]["ep_total_reward"] = float(self._ep_total_reward)
        return obs, reward, terminated, truncated, info

