# Rat-robot test

Timing test and trajectory following test for robot in Mujoco



# Getting started
### Environment: 
System with Python 3 (Personal recommendation Ubutnu 20.04 or Ubuntu 22.04)

### Dependencies
Package: <strong>matplotlib</strong>  and <strong>numpy</strong>
 
Install command: `pip3 install <package name>`

### Mujoco 
- Version: 2.1.0 in https://github.com/deepmind/mujoco/releases/tag/2.1.0
- Python package: Mujoco_py in https://github.com/openai/mujoco-py
- Install process: please follow the readme in https://github.com/openai/mujoco-py

# Run experiments

There are three test cases in the folder <strong>v1- SoftControl</strong>
- CalTime: to check the detail timing cost of two model
- Swing: to let the robot hang in the air, and check leg trajectories following of two models
- Walking: to ask the robot go straight with the ideal trot gait

File structure in each case is
- Test case:
    - models: the folder to store the mujoco model in XML
    - src: the folder to store the running code that based on python 3

Running simulation in each case:
- Simulation of Alex's model:
    - Description: the mouse control model is designed with inverse kinematics
    - Running command: please use `python3 motion_module.py` in <strong>Alex_src</strong>
- Simulation of Yuhong's model,
    - Description: the mouse control model is designed with geometric modeling
    - Running command: please use `python3 sim_test.py` in <strong>Yuhong_src</strong>

For test cases <strong>Swing</strong> and <strong>Walking</strong>, it has a parameter `fre` which overrides the length of one gait stride. It can be set in the simulation starting with `python3 xx.py --fre=<>`, or directly be modified in the main function of the execution script.

For <strong>Controller</strong>, it hase a high level controller that use PS4 to control the robot walking.

# Nermo
The details is shown in the folder "Mouse_SRC_1"
 
