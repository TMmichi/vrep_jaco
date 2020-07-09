# vrep_jaco

IITP Anthrophomorphic Robot Arm & Hand Project

Manipulator controlling ROS package with moveit! for V-rep simulation / Local kinova jaco2 machine

Currently tested in Ubuntu 18.04 and ROS melodic

## 1. Installation

Installation in Ubuntu 18.04 with ROS melodic is recommended (since other version of ROS or ubuntu distro were not tested)

### 1-1. Preliminary

#### 1-1-1. V-rep (Coppelia) installation

V-rep source can be downloaded from [here](http://www.coppeliarobotics.com/ubuntuVersions.html) and should be installed within the `/opt` folder. Installed location can be varied, but should be matched with the vrep_path argument within the launch file: `vrep_jaco_bringup/launch/bringup.launch: vrep_path`
If have you install Coppelia (v4.0.0) instead of V-rep (v3.6.n), you need to move file:`libsimExtROSInterface.so` from folder:`compiledRosPlugins` to the source folder.

Within the source file:
```bash
cd compiledRosPlugins
mv libsimExtROSInterface.so ..

```

#### 1-1-2. Create workspace with individual folders for vrep_jaco and moveit

Since we are not using official repo of moveit, it is required to create and build moveit from the modified source of our repo.

```bash
mkdir -p ~/name_of_your_workspace/moveit ~/name_of_your_workspace/vrep_jaco
```

#### 1-1-3. Create src folder in moveit directory with wstool

Because vrep_jaco package requires moveit package to be overlayed, building and sourcing prior to making vrep_jaco is required. This folder will be later used to store the repo of moveit source and the vrep_jaco repo along side.
Within your moveit folder in workspace,

```bash
cd ~/name_of_your_workspace/moveit
wstool init src
```

to create `src` folder with .rosinstall file in it.

#### 1-1-4. ROS installation

It is assumed that the one is capable of installing ROS with full compatibility. This package is running python3, so proper steps to deal with python3 should be followed, Additionally, standalone `catkin` package is also required, so it should be installed along with the ROS default `catkin_make`

```bash
sudo apt-get install python3-pip python3-yaml
sudo pip3 install rospkg catkin_pkg

sudo apt-get install python-catkin-tools    // if using Ubuntu
sudo pip install -U catkin_tools            // if using other OS
```

### 1-2. Clone and install dependencies, build & source repo for Moveit installation

Within your moveit folder, (the directory where your `src` folder is at)

```bash
wstool merge -t src https://raw.githubusercontent.com/TMmichi/moveit/master/vrep_jaco_moveit.rosinstall
wstool update -t src
rosdep install -y --from-paths src --ignore-src --rosdistro ${ROS_DISTRO}
catkin config --extend /opt/ros/${ROS_DISTRO} --cmake-args -DCMAKE_BUILD_TYPE=Release
sudo apt-get install libusb-dev libbluetooth-dev libcwiid1 libcwiid-dev
catkin build
```

Source `setup.bash` file in your project devel folder to the `.bashrc`.

```bash
source ~/name_of_your_workspace/moveit/devel/setup.bash
```

### 1-3. Remaining packages installation

Within the moveit package, ompl package and ik algorithm from TRAC is used, and hence should be installed properly.

```bash
sudo apt-get install ros-${ROS_DISTRO}-ompl
sudo apt-get install ros-${ROS_DISTRO}-trac-ik-kinematics-plugin
```

##### Note 1

- If your libqt5x11extras5 version is greater than 5.5.1-3build1, you should downgrade your libqt with command

```bash
sudo apt-get install libqt5x11extras5=5.5.1-3build1
```

  in order to install ros-<distro>-rviz-visual-tools within the preliminaries: `vrep_jaco_moveit.rosinstall`.

### 1-4. Clone vrep_jaco, build & source

(MANDATOROY: YOU MUST SOURCE `setup.bash` IN MOVEIT DEVEL FOLDER)
After sourcing the `setup.bash` file within moveit to overlay, go to your vrep_jaco folder in your workspace, and clone package.

```bash
cd ~/name_of_your_workspace/vrep_jaco
git clone https://github.com/TMmichi/vrep_jaco.git
mv vrep_jaco src
```

It is required to have all of your sources in a folder name `src`, so please change `vrep_jaco` folder with the command above.

Build your repo and source it. With the following command, sourcing will done automatically within the terminal at launching.

```bash
catkin_make
echo "source ~/name_of_your_workspace/vrep_jaco/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 1-5 ROS package modification

Since we are teaking ROS timer with sim_time, it is required to modify `timer.py` in ROS package. Comment line 163, where it says `raise rospy.exceptions.ROSTimeMobedBackwardsException(time_jump)`. This is necessary because we are constantly reseting our ROS timer at the reset call of each episodes. You can find `timer.py` file with the following command below.

```bash
cd /opt/ros/${ROS_DISTRO}/lib/python2.7/dist-packages/rospy
```

### 1.6 Post installation
```
pip install pyyaml matplotlib
```

## 2. Usage (WIP)

Manipulation of a real machine and one in the simulation are much alike from each other.

### 2-1. Manipulator control within the V-rep simulation

#### 2-1-1. Simulation environment bringup

```bash
roslaunch vrep_jaco_bringup bringup.launch
```

`bringup.launch` file will launch a V-rep env with the scene file `jaco_table2.ttt`
Bringup launch file DOES NOT include manipulator URDF xacro.

#### 2-1-2. Vrep client side initialization with RL Environment

```bash
roslaunch rl_controller rl_controller.launch
```

##### Note 2

- Prior to connect ROS-api of V-rep with the simulation environment itself, V-rep should be launched with the designated port number. Default port number has been set to `19997` in `rl_controller/scripts/env/env_vrep_client.py` line number `15` with `server_port`.

- If the api client node is running on a separate machine other than the machine with V-rep simulation, IP address should also be clarified, other than the default localhost (`127.0.0.1`) in `rl_controller/scripts/env/env_vrep_client.py` line number `14` with `server_address`.

#### 2-1-3. Manipulator control node with moveit! in ROS node

```bash
roslaunch jaco_controller_integrated jaco_controller_integrated.launch
```

`jaco_controller_integrated.launch` includes most of the parameters required from the moveit! package with control parameters and launches the visualization node with RVIZ.

It also launches the ROS node with C++ script `jaco_controller_integrated.cpp` in folder `jaco_controller_integrated/src` which initializes the actionlib client side which communicates with the v-rep api server side.

By using Moveit! package, user does not have to consider action/state synchronization between the machine in the simulation and the controller node. Providing target gripper pose to the `plan()` method within the `move_group` instance will solve inverse kinematics problem for designated pose and store the solution as a member variable.
Calling `sendgoal()` method within the action_client will publish rostopic with name and type in `/USER_DEFINED_NAME/follow_joint_trajectory/goal` and `control_msgs/FollowJointTrajectoryActionGoal` respectively. Published topic from move_gropup instance in actionlib client side will then be transfered to the server side and wait till all of the goal joint states in `follow_joint_trajectory/goal` to be finished in the simulation.

### 2-2. Manipulator control of a Real Machine

#### 2-2-1. Jaco  bringup

```bash
roslaunch kinova_bringup kinova_robot.launch
```

#### 2-2-2. Manipulator control node with moveit! in ROS node (real machine)

```bash
roslaunch jaco_controller_kinova jaco_controller_kinova.launch
```

As it is mentioned before, there is no significant differnece with the simulation ros control script.

#### 2-2-3. Integrated Control - Simulation / Real Machine

`Added`: `jaco_controller_integrated.launch` will take over both of the simulation side and the real side in control. (WIP)

## 3. Contents

Consists:

```list
- action_client
- jaco_arm_pkgs
- jaco_controller_integrated
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

`jaco_controller_integrated` package is a C++ vrep simulated controller script named 'jaco_controller.cpp' loacted in src folder. It uses moveit! for the low-level control. By providing target pose of the gripper on line 78 it planns its trajectory and velocity profile, and on line 84, it moves as its given profile.

`kinova-ros` package is a prerequisite for the real jaco.

`moveit` is the moveit! package, and shouldn't be eddited.

`rl_controller` is the main work scheme script, where it encodes states, produce action from policy, and trains via RL and IRL.

Should build ROS package for python3 in order to use moveit!. Follow instructions given in hyperlink: 
[Instructions](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674)

#### Reference

[kinova-ros_source_code](https://github.com/Kinovarobotics/kinova-ros.git) from github@kinova-robotics

[V-rep_api](https://github.com/JoshSong/jaco_ros_vrep.git) from github@JoshSong

[controller_script](http://docs.ros.org/kinetic/api/moveit_tutorials/html/index.html) from moveit_tutorial
