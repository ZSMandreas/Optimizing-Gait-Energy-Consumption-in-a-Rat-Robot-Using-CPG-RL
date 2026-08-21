from stable_baselines3.common.save_util import load_from_zip_file

data, params, pytorch_vars = load_from_zip_file("/home/simiao/rat_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models/checkpoints/rat_mapping_500000_steps.zip")
print("sb3_version in model:", data.get("sb3_version"))
print("python_version in model:", data.get("python_version"))
