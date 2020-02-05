# vrep_jaco

IITP Anthrophomorphic Robot Arm & Hand Project



Consists:
```
- action_client
- jaco_arm_pkgs
- jaco_controller
- jaco_controller_kinova
- kinova-ros
- moveit
- rl_controller
- vrep_api
- vrep_jaco_bringup
- vrep_jaco_data
- vrep_jaco_moveit
```

`action_client` package wraps vrep environment with C++ client. It uses `vrep_api` package

`jaco_arm_pkgs` package provides URDF information of jaco robot for the moveit! to consider

`jaco_controller` package is a C++ vrep simulated controller script named 'jaco_controller.cpp' loacted in src folder. It uses moveit! for the low-level control. By providing target pose of the gripper on line 78 it planns its trajectory and velocity profile, and on line 84, it moves as its given profile.

`jaco_controller_kinova` package works same as `jaco_controller`, but it controlls the real robot

`kinova-ros` package is a prerequisite for the real jaco.

`moveit` is the moveit! package, and shouldn't be eddited.

`rl_controller` is the main work scheme script, where it encodes states, produce action from policy, and trains via RL and IRL.

Should build ROS package for python3 in order to use moveit!. Follow instructions given URL below.
Instructions: <https://www.miguelalonsojr.com/blog/robotics/ros/python3/2019/08/20/ros-melodic-python-3-build.html>
