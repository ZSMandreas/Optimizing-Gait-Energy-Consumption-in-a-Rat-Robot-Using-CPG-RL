# -*- coding: utf-8 -*-
"""
This script evaluates a previously trained quadruped walking policy and augments
the low–level control with a simple open‑loop spine actuator motion.  It is
designed to test whether the existing four‑leg model can maintain stable,
straight‑line locomotion when a small periodic movement is introduced on the
spine joint.  The base locomotion logic for the legs remains unchanged: the
latent two‑dimensional commands per leg are passed through the StarMapper to
produce foot positions and then inverse‑mapped into joint angles.  Only the
spine control is injected here in a feedforward manner.

Usage example:

    python Rat_Eval_with_spine.py \
        --ckpt_dir ./checkpoints \
        --prefix rat_mapping \
        --amplitude 0.1 \
        --period 1.0 \
        --eval_steps 1000 \
        --seed 123

Command line arguments allow selection of the trained model, the vector
normalization file, and the parameters of the spine motion (amplitude and
period).  See `--help` for full details.

Note: This script assumes that the underlying MuJoCo model defines a spine
actuator immediately following the eight leg actuators.  If your model
arranges actuators differently, adjust the `SPINE_ACT_IDX` constant below to
match the index of the spine actuator in `data.ctrl`.
"""

import argparse
import glob
import math
import os
os.chdir("/home/geriatronics/ratmujoco_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models")
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Import the original environment.  Do not modify this class – all leg
# behaviour remains as in the trained model.
from Rat_Env_duty import Go2Env


def find_latest_checkpoint(root: str, prefix: str):
    """Locate the most recent checkpoint and its associated VecNormalize file.

    Parameters
    ----------
    root : str
        Directory containing checkpoint files.
    prefix : str
        Filename prefix used for checkpoints (e.g. "rat_mapping").

    Returns
    -------
    tuple(str, str)
        A tuple of (model_path, vecnorm_path).  If no checkpoints are found,
        both values are ``None``.
    """
    zips = sorted(glob.glob(os.path.join(root, f"{prefix}_*_steps.zip")))
    if not zips:
        return None, None
    latest_zip = zips[-1]
    step_str = latest_zip.split("_")[-2]  # e.g. rat_mapping_3000000_steps.zip → "3000000"
    vn_path = os.path.join(root, f"{prefix}_{step_str}_vecnormalize.pkl")
    if not os.path.exists(vn_path):
        vn_path = None
    return latest_zip, vn_path


def extract_raw_env(env):
    """Recursively unwrap a vectorized or normalized environment to obtain the base env.

    This helper walks through .venv and .env attributes until it reaches the
    underlying Go2Env instance.  It is necessary because stable_baselines3
    wraps environments for normalization and vectorization.

    Parameters
    ----------
    env : VecEnv or gym.Env
        The environment returned by `make_vec_env` or `VecNormalize`.

    Returns
    -------
    gym.Env
        The unwrapped Go2Env instance.
    """
    # stable_baselines3 VecNormalize wraps a VecEnv under .venv
    if hasattr(env, 'venv'):
        env = env.venv
    # VecEnv exposes envs list
    if hasattr(env, 'envs'):
        # take the first env in the vector
        env = env.envs[0]
    # Some wrappers further nest an 'env' attribute
    while hasattr(env, 'env'):
        env = env.env
    return env


def main(args):
    # Resolve model and vector norm paths
    model_path = args.model
    vn_path = args.vecnorm
    if model_path is None:
        model_path, vn_path_auto = find_latest_checkpoint(args.ckpt_dir, args.prefix)
        if model_path is None:
            # Fall back to final checkpoint
            model_path = os.path.join(args.ckpt_dir, f"{args.prefix}_final.zip")
            vn_path_auto = os.path.join(args.ckpt_dir, f"{args.prefix}_final_vecnormalize.pkl")
            print("[WARN] No step checkpoint found; falling back to FINAL.")
        if vn_path is None:
            vn_path = vn_path_auto

    # Construct evaluation environment.  We keep a single environment instance
    # because the spine control is applied deterministically to a single robot.
    vec_env = make_vec_env(lambda: Go2Env(render_mode="window"), n_envs=1, seed=args.seed)

    # Load or create VecNormalize
    if vn_path is not None and os.path.exists(vn_path):
        vec_env = VecNormalize.load(vn_path, vec_env)
        print(f"[INFO] Loaded VecNormalize statistics from: {vn_path}")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True)
        print("[INFO] No VecNormalize file found. Created a fresh one (results may differ from training).")
    vec_env.training = False
    vec_env.norm_reward = False

    # Load policy
    model = PPO.load(model_path, env=vec_env, device=args.device)
    print(f"[INFO] Loaded trained model: {model_path}")

    # Extract underlying Go2Env to access mujoco data for spine control
    raw_env = extract_raw_env(vec_env)

    # Determine index of the spine actuator in Mujoco control vector.  By
    # convention in the provided rat model, actuators 0–7 correspond to the
    # eight leg joints and the ninth actuator controls the spine.  Adjust
    # SPINE_ACT_IDX if your model orders actuators differently.
    SPINE_ACT_IDX = args.spine_act_index
    ctrl_dim = raw_env.data.ctrl.shape[0]
    if SPINE_ACT_IDX < 0 or SPINE_ACT_IDX >= ctrl_dim:
        raise ValueError(f"Invalid spine actuator index {SPINE_ACT_IDX}. Model has {ctrl_dim} actuators.")

    # Initialise time accumulator for the spine motion.  Each environment step
    # internally advances the simulation by raw_env.n * raw_env.dt seconds.
    sim_dt = getattr(raw_env, 'dt', 0.01) * getattr(raw_env, 'n', 1)
    phase_time = 0.0

    # Reset environment and begin evaluation loop
    obs = vec_env.reset()
    step_count = 0
    # Precompute angular frequency
    omega = 2 * math.pi / args.period if args.period > 0 else 0.0
    while step_count < args.eval_steps:
        # Compute feedforward spine command using a simple sinusoid.  The
        # amplitude and period are configurable via command line.  If period is
        # zero, spine remains at zero.
        spine_value = args.amplitude * math.sin(omega * phase_time) if args.period > 0 else 0.0
        # Inject spine control into the mujoco control vector.  Do this
        # immediately before the call to step() so that it is applied during
        # the simulation integration for this action.
        raw_env.data.ctrl[SPINE_ACT_IDX] = float(spine_value)

        # Query the trained policy for leg actions (8 dims) and step the env
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        # Update time and counters.  The time increment uses the underlying
        # simulation time step multiplied by the number of internal repeats per
        # environment step.
        phase_time += sim_dt
        step_count += 1

        # Render on each step if requested
        
        vec_env.render()

        # Reset environment on episode termination
        if done:
            obs = vec_env.reset()
            phase_time = 0.0

    # Clean up
    vec_env.close()
    print("[DONE] Spine‑augmented evaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Directory containing checkpoints")
    parser.add_argument("--prefix", type=str, default="rat_mapping", help="Checkpoint filename prefix")
    parser.add_argument("--model", type=str, default=None, help="(Optional) specific model .zip file to load")
    parser.add_argument("--vecnorm", type=str, default=None, help="(Optional) specific VecNormalize .pkl to load")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of environment steps to evaluate")
    parser.add_argument("--seed", type=int, default=999, help="Random seed for environment initialisation")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device to run the policy on")
    parser.add_argument("--amplitude", type=float, default=0.5, help="Amplitude of the spine sinusoid (radians)")
    parser.add_argument("--period", type=float, default=1.0, help="Period of the spine sinusoid (seconds). Use 0 to disable motion.")
    parser.add_argument("--spine_act_index", type=int, default=8, help="Index of the spine actuator in the Mujoco control vector")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    args = parser.parse_args()
    main(args)