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
from pyrep import PyRep
from assets.jaco import jaco
from std_msgs.msg import Int8, Int8MultiArray, Float32MultiArray
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory


def radtoangle(rad):
    return rad / pi * 180


class JacoVrepEnvUtil(object):
    def __init__(self, **kwargs):
        #self.debug = kwargs['debug']

        ### ------------  PyRep INITIALIZATION  ------------ ###
        #self.scene = rospy.get_param("/rl_controller/scene_file")
        self.scene = "/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/vrep_jaco_bringup/scene/jaco_table_simple.ttt"
        self.pr = PyRep()
        self.pr.launch(self.scene, headless=False)
        time.sleep(3)
        self.arm = jaco()

        ### ------------  JOINT HANDLES INITIALIZATION  ------------ ###
        self.jointState_ = JointState()
        self.jointState_.position = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.base_position = []
        # Joint prefix setup
        vrepArmPrefix = "jaco_joint"
        vrepFingerPrefix = "jaco_joint_finger"
        vrepFingerTipPrefix = "jaco_joint_finger_tip"
        urdfArmPrefix = "j2n6s300_joint_"
        urdfFingerPrefix = "j2n6s300_joint_finger_"
        urdfFingerTipPrefix = "j2n6s300_joint_finger_tip_"
        # Handle init
        self._initJoints(
            vrepArmPrefix, vrepFingerPrefix, vrepFingerTipPrefix,
            urdfArmPrefix, urdfFingerPrefix, urdfFingerTipPrefix)
        self.target_angle = [0,0,0,0,0,0]
        self.gripper_pose = []
        self.gripper_angle_1 = 0.35    # finger 1, 2
        self.gripper_angle_2 = 0.35    # finger 3

        ### ------------  ROS INITIALIZATION  ------------ ###
        #self.rate = kwargs['rate']
        self.rate = 50  #TODO remove
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
            "rl_action_output", Float32MultiArray, queue_size=1)
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
        self.jointState_.position = self.arm.get_joint_positions() # TODO: Find out what's the unit for positions

    def _jointState_CB(self, msg):
        self.jointState_.position = msg.position

    def _trajCB_raw(self, msg):
        points = msg.points[-1]
        self.target_angle = points.positions[:6]
        position = self.jointState_.position
        try:
            move_diff = np.linalg.norm(
                np.array(points.positions[:6])-np.array(position[:6]))
            if (not move_diff > 1) or (not move_diff < 6):
                # TODO: Find out what's the unit for positions / do we need to 
                self.arm.set_joint_target_positions([radtoangle(-points.positions[i]) for i in range(6)])
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
            self.pr.step()

    def _reset(self, target_angle=None, sync=False):
        self.gripper_angle_1 = 0.35
        self.gripper_angle_2 = 0.35
        proc_reset = self._memory_check()
        self._vrep_process_reset() if proc_reset else None
        self.pr.stop()
        if target_angle == None:
            random_init_angle = [-1*sample(range(-180, 180), 1)[0], -150, -1*sample(range(200, 270), 1)[0], -1*sample(
                range(50, 130), 1)[0], -1*sample(range(50, 130), 1)[0], -1*sample(range(50, 130), 1)[0]]
        else:
            random_init_angle = target_angle
        self.arm.set_joint_positions(random_init_angle)
        self.arm.set_joint_target_positions(random_init_angle)
        self.pr.start()
        for _ in range(10):
            self.pr.step()
        obs = self._get_observation()[0]
        self.goal = self._sample_goal()
        dist_diff = np.linalg.norm(
            np.array(obs[:3]) - np.array(self.goal))
        self.base_position = np.array(self.obj_get_position(self.jointHandles_[0]))
        self.ref_reward = (3 - dist_diff*1.3)
        self.reset_pub.publish(Int8(data=ord('r')))
        time.sleep(1)
        return obs

    def _memory_check(self):
        total = psutil.virtual_memory().total
        used = total - psutil.virtual_memory().available
        return True if (used/total*100) > 75 else False

    def _vrep_process_reset(self):
        print("Restarting Vrep")
        self.worker_pause = True
        self.pr.stop()
        self.pr.shutdown()
        self.pr.launch(self.scene, headless=False)
        time.sleep(3)
        self.worker_pause = False

    def _get_observation(self,):
        # TODO: Use multiprocessing to generate state from parallel computation
        test = True  # TODO: Remove test
        if test:
            self.gripper_pose = self.obj_get_position(self.jointHandles_[5])
            observation = self.gripper_pose + self.obj_get_orientation(self.jointHandles_[5])
            #observation += self.jointState_.position[:6]
            #for i_jointhandle in self.jointHandles_[:6]:
            #    observation.append(self.obj_get_joint_angle(i_jointhandle))
            observation += self.goal
            #if np.isnan(np.sum(observation)):
                #print("NAN OCCURED IN VREP CLIENT!!!!!")
                # If nan, try to get observation recursively untill we do not have any nan
            #    observation = self._get_observation()[0]
        else:
            data_from_callback = []
            observation = self.state_gen.generate(data_from_callback)
        return np.array(observation), self.target_angle

    def _get_reward(self):
        # TODO: Reward from IRL
        gripper_pose = np.array(self.gripper_pose)
        if self.reward_method == "l2":
            wb = np.linalg.norm(gripper_pose - self.base_position)
            if wb < 0.85:
                if 3.14 - 0.15 < self.jointState_.position[2] < 3.14 + 0.15:
                    reward = -1
                else:
                    dist_diff = np.linalg.norm(gripper_pose - np.array(self.goal))
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
            np.array(self.gripper_pose) - np.array(self.goal))
        if dist_diff < 0.15:  # TODO: Shape terminal inspection
            print("Target Reached")
            return True, 100
        else:
            return False, 0

    def _take_action(self, a):
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

if __name__ == "__main__":
    try:
        env_test_class=JacoVrepEnvUtil()
        rospy.spin()
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")
