#!/usr/bin/env python

import os
import sys
import time
import datetime
import psutil
import signal
import subprocess
from math import pi
from random import sample, randint, uniform

import numpy as np

import rospy
from env.vrep_env_rl import vrep_env
from env.vrep_env_rl import vrep  # vrep.sim_handle_parent
from std_msgs.msg import Int8, Int8MultiArray, Float32MultiArray
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory


def radtoangle(rad):
    return rad / pi * 180


class JacoVrepEnvUtil(vrep_env.VrepEnv):
    def __init__(self, **kwargs):
        self.debug = kwargs['debug']

        ### ------------  V-REP API INITIALIZATION  ------------ ###
        vrep_exec = rospy.get_param(
            "/rl_controller/vrep_path")+"/coppeliaSim.sh -s "
        scene = rospy.get_param("/rl_controller/scene_file")
        self.exec_string = vrep_exec+scene+" &"
        self.addr = kwargs['server_addr']
        self.port = kwargs['server_port']

        subprocess.call(self.exec_string, shell=True)
        time.sleep(3)
        vrep_env.VrepEnv.__init__(
            self, self.addr, self.port)

        ### ------------  JOINT HANDLES INITIALIZATION  ------------ ###
        self.jointState_ = JointState()
        self.jointState_.position = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.jointHandles_ = []
        self.base_position = []
        # Joint prefix setup
        vrepArmPrefix = "jaco_joint"
        vrepFingerPrefix = "jaco_joint_finger"
        vrepFingerTipPrefix = "jaco_joint_finger_tip"
        urdfArmPrefix = "j2n6s300_joint_"
        urdfFingerPrefix = "j2n6s300_joint_finger_"
        urdfFingerTipPrefix = "j2n6s300_joint_finger_tip_"
        # Handle init
        self.jointHandles_ = self._initJoints(
            vrepArmPrefix, vrepFingerPrefix, vrepFingerTipPrefix,
            urdfArmPrefix, urdfFingerPrefix, urdfFingerTipPrefix)
        self.base_position = np.array(self.obj_get_position(self.jointHandles_[0]))
        self.target_angle = [0,0,0,0,0,0]
        self.gripper_pose = []
        self.gripper_angle_1 = 0.35    # finger 1, 2
        self.gripper_angle_2 = 0.35    # finger 3

        ### ------------  ROS INITIALIZATION  ------------ ###
        self.rate = kwargs['rate']
        # Subscribers / Publishers
        self.key_sub = rospy.Subscriber(
            "key_input", Int8, self._keys, queue_size=10)
        self.joint_state_sub = rospy.Subscriber(
            "/j2n6s300/joint_states", JointState, self._jointState_CB, queue_size=10)
        self.rs_image_sub = rospy.Subscriber(
            "/vrep/depth_image", Image, self._depth_CB, queue_size=1)
        self.pressure_sub = rospy.Subscriber(
            "/vrep/pressure_data", Float32MultiArray, self._pressure_CB, queue_size=1)
        self.traj_sub = rospy.Subscriber(
            "j2n6s300/trajectory", JointTrajectory, self._trajCB_raw, queue_size=1)
        self.reset_pub = rospy.Publisher("reset_key", Int8, queue_size=1)
        self.quit_pub = rospy.Publisher("quit_key", Int8, queue_size=1)
        self.action_pub = rospy.Publisher(
            "rl_action_output", Float32MultiArray, queue_size=10)
        self.target_pub_ = rospy.Publisher(
            "test_target", Float32MultiArray, queue_size=1)
        self.worker_pause = False
        self.action_received = False

        ### ------------  STATE GENERATION  ------------ ###
        try:
            self.state_gen = kwargs['stateGen']
        except Exception:
            self.state_gen = None
        self.image_buffersize = 5
        self.image_buff = []
        self.pressure_buffersize = 100
        self.pressure_state = []

        self.depth_trigger = True
        self.pressure_trigger = True
        self.data_buff = []
        self.data_buff_temp = [0, 0, 0]

        ### ------------  REWARD  ------------ ###
        self.goal = self._sample_goal()
        try:
            self.reward_method = kwargs['reward_method']
            self.reward_module = kwargs['reward_module']
        except Exception:
            self.reward_method = None
            self.reward_module = None


    def _initJoints(
            self,
            vrepArmPrefix, vrepFingerPrefix, vrepFingerTipPrefix,
            urdfArmPrefix, urdfFingerPrefix, urdfFingerTipPrefix):
        """Initialize joints object handles and joint states
        """
        in_names = []
        for i in range(1, 7):
            in_names.append(vrepArmPrefix+str(i))
            outname = urdfArmPrefix+str(i)
            self.jointState_.name.append(outname)
            self.jointState_.velocity.append(0)
            self.jointState_.effort.append(0)
        for i in range(1, 4):
            in_names.append(vrepFingerPrefix+str(i))
            outname = urdfFingerPrefix+str(i)
            self.jointState_.name.append(outname)
            self.jointState_.velocity.append(0)
            self.jointState_.effort.append(0)
        for i in range(1, 4):
            in_names.append(vrepFingerTipPrefix+str(i))
            outname = urdfFingerTipPrefix+str(i)
            self.jointState_.name.append(outname)
            self.jointState_.velocity.append(0)
            self.jointState_.effort.append(0)
        jointHandles_ = list(map(self.get_object_handle, in_names))
        self.jointState_.position = list(
            map(self.obj_get_joint_angle, jointHandles_))
        return jointHandles_

    def _jointState_CB(self, msg):
        if not (True in np.isnan(msg.position)):
            self.jointState_.position = msg.position
        else:
            pass


    def _trajCB_raw(self, msg):
        #print("traj received from moveit at: ", datetime.datetime.now())
        points = msg.points[-1]
        self.target_angle = points.positions[:6]
        position = self.jointState_.position
        try:
            move_diff = np.linalg.norm(
                np.array(points.positions[:6])-np.array(position[:6]))
            if (not move_diff > 1) or (not move_diff < 6):
                for j in range(0, 6):
                    self.obj_set_position_target(
                        self.jointHandles_[j], radtoangle(-points.positions[j]))
        except Exception as e:
            print(e, file=sys.stderr)
        self.action_received = True

    def _keys(self, msg):
        self.key_input = msg.data
        #print("INPUT: ",self.key_input)
        if self.key_input == ord('r'):      # Reset environment
            self._reset()
            self.key_input = ord('3')
        elif self.key_input in [ord('f'), ord('g'), ord('v'), ord('b'), ord('o'), ord('p')]:
            self._take_manual_action(self.key_input)
        elif self.key_input == ord('2'):
            self.step_simulation()

    def _reset(self, target_angle=None, sync=False):
        #print("Entered reset at: ",rospy.Time.now())
        self.gripper_angle_1 = 0.35
        self.gripper_angle_2 = 0.35
        proc_reset = self._memory_check()
        self._vrep_process_reset() if proc_reset else None
        if self.sim_running:
            self.stop_simulation()
        if target_angle == None:
            random_init_angle = [sample(range(-180, 180), 1)[0], 150, sample(range(200, 270), 1)[0], sample(
                range(50, 130), 1)[0], sample(range(50, 130), 1)[0], sample(range(50, 130), 1)[0]]
        else:
            random_init_angle = target_angle
        for i, degree in enumerate(random_init_angle):
            self.obj_set_position_inst(self.jointHandles_[i], -degree)
            self.obj_set_position_target(self.jointHandles_[i], -degree)
        #print("Resume sim at: ",rospy.Time.now())
        self.start_simulation(sync=sync, time_step=0.05)
        for _ in range(10):
            self.step_simulation()
        #time.sleep(1)       # time required for V-rep to reset its timer 
        obs = self._get_observation()[0]
        self.goal = self._sample_goal()
        dist_diff = np.linalg.norm(
            np.array(obs[:3]) - np.array(self.goal))
        self.ref_reward = (3 - dist_diff*1.3)
        #print("Send reset signal to moveit at: ",rospy.Time.now())
        self.reset_pub.publish(Int8(data=ord('r')))
        time.sleep(1)     # time required for moveit to be ready / topics sent to moveit to be flushed
        return obs

    def _memory_check(self):
        total = psutil.virtual_memory().total
        used = total - psutil.virtual_memory().available
        return True if (used/total*100) > 75 else False

    def _vrep_process_reset(self):
        print("Restarting Vrep")
        self.worker_pause = True
        self.disconnect()
        quit_signal = Int8()
        quit_signal.data = 1
        self.quit_pub.publish(quit_signal)
        time.sleep(8)
        subprocess.call(self.exec_string, shell=True)
        time.sleep(3)
        self.connect(self.addr, self.port)
        time.sleep(1)
        self.worker_pause = False

    def _get_observation(self):
        # TODO: Use multiprocessing to generate state from parallel computation
        test = True  # TODO: Remove test
        if test:
            self.gripper_pose = self._get_gripper_position() + self._get_gripper_orientation()
            observation = self.gripper_pose + self.goal
        else:
            data_from_callback = []
            observation = self.state_gen.generate(data_from_callback)
        return np.array(observation), self.target_angle
    
    def _get_gripper_position(self):
        gripper_position = self.obj_get_position(self.jointHandles_[5])
        if not (True in np.isnan(gripper_position)):
            return gripper_position
        else:
            return self._get_gripper_position()
    
    def _get_gripper_orientation(self):
        gripper_orientation = self.obj_get_orientation(self.jointHandles_[5])
        if not (True in np.isnan(gripper_orientation)):
            return gripper_orientation
        else:
            return self._get_gripper_orientation()

    def _get_reward(self):
        # TODO: Reward from IRL
        gripper_pose = np.array(self.gripper_pose)
        if self.reward_method == "l2":
            wb = np.linalg.norm(gripper_pose[:3] - self.base_position)
            if wb < 0.85:
                if 3.14 - 0.15 < self.jointState_.position[2] < 3.14 + 0.15:
                    reward = -1
                else:
                    dist_diff = np.linalg.norm(gripper_pose[:3] - np.array(self.goal))
                    reward = ((3 - dist_diff*1.3) - self.ref_reward) * 0.1  # TODO: Shape reward
            else:
                reward = -1
            return reward
        elif self.reward_method == "":
            return self.reward_module(gripper_pose, self.goal)
        else:
            print("\033[31mConstant Reward. SHOULD BE FIXED\033[0m")
            return 30

    def _sample_goal(self):
        target_pose = [uniform(0.2, 0.5) * sample([-1, 1], 1)[0]
                       for i in range(2)] + [uniform(0.8, 1.1)]
        target_out = Float32MultiArray()
        target_out.data = np.array(target_pose, dtype=np.float32)
        self.target_pub_.publish(target_out)
        return target_pose

    def _get_terminal_inspection(self):
        dist_diff = np.linalg.norm(
            np.array(self.gripper_pose[:3]) - np.array(self.goal))
        if dist_diff < 0.15:  # TODO: Shape terminal inspection
            print("Target Reached")
            return True, 100
        else:
            return False, 0

    def _take_action(self, a):
        #print(a,rospy.Time.now().to_sec())
        key_out = Float32MultiArray()  # a = [-1,0,1] * 8
        key_out.data = np.array(a[:6], dtype=np.float32)
        self.action_pub.publish(key_out)
        self.gripper_angle_1 = max(
            min(self.gripper_angle_1 + a[6]/20.0, 0), -0.7)
        self.gripper_angle_2 = max(
            min(self.gripper_angle_2 + a[7]/20.0, 0), -0.7)
        self.obj_set_position_target(
            self.jointHandles_[6], radtoangle(self.gripper_angle_1))
        self.obj_set_position_target(
            self.jointHandles_[7], radtoangle(self.gripper_angle_1))
        self.obj_set_position_target(
            self.jointHandles_[8], radtoangle(self.gripper_angle_2))

    def _take_manual_action(self, key):
        if key == ord("f"):
            inc1 = 0.05
            inc2 = 0
        elif key == ord("g"):
            inc1 = -0.05
            inc2 = 0
        elif key == ord("v"):
            inc1 = 0
            inc2 = 0.05
        elif key == ord("b"):
            inc1 = 0
            inc2 = -0.05
        elif key == ord("o"):
            inc1 = 0.05
            inc2 = 0.05
        elif key == ord("p"):
            inc1 = -0.05
            inc2 = -0.05
        else:
            pass
        self.gripper_angle_1 = max(min(self.gripper_angle_1+inc1, 0), -0.7)
        self.gripper_angle_2 = max(min(self.gripper_angle_2+inc2, 0), -0.7)
        for i in range(6, 8):
            self.obj_set_position_target(
                self.jointHandles_[i], radtoangle(self.gripper_angle_1))
        self.obj_set_position_target(
            self.jointHandles_[8], radtoangle(self.gripper_angle_2))

    # TODO: data saving method
    def _depth_CB(self, msg):
        self.depth_trigger = True
        self.pressure_trigger = True

        msg_time = round(msg.header.stamp.to_sec(), 2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data, dtype=np.uint16)
        data = np.reshape(data, (height, width))
        data = np.flip(data, 0)
        self.image_buff = [data, msg_time]
        try:
            self.data_buff_temp[0] = self.image_buff
        except Exception:
            pass

    def _pressure_CB(self, msg):
        try:
            if self.pressure_trigger:
                msg_time = round(msg.data[0], 2)
                self.pressure_state.append([msg.data[1:], msg_time])
                if len(self.pressure_state) > self.pressure_buffersize:
                    self.pressure_state.pop(0)
                #print("pressure state: ", msg_time)
                self.data_buff_temp[2] = self.pressure_state[-1]
                self.pressure_trigger = False
        except Exception:
            pass
