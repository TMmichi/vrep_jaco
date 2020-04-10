#!/usr/bin/env python

from env.vrep_env_rl import vrep_env
from env.vrep_env_rl import vrep  # vrep.sim_handle_parent

import os
import sys
import time
import psutil, signal
import subprocess
from math import pi
from random import sample, randint

import numpy as np
from matplotlib import pyplot as plt

import rospy
from env.SimpleActionServer_mod import SimpleActionServer_mod
from std_msgs.msg import Int8
from std_msgs.msg import Int8MultiArray
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryFeedback
from control_msgs.msg import FollowJointTrajectoryResult


def radtoangle(rad):
    return rad / pi * 180


class JacoVrepEnvUtil(vrep_env.VrepEnv):
    def __init__(self, **kwargs):
        ### ------------  V-REP API INITIALIZATION  ------------ ###
        vrep_exec = rospy.get_param("/vrep_jaco_bringup_simulator/vrep_path")+"/coppeliaSim.sh -s "
        scene = rospy.get_param("/vrep_jaco_bringup_simulator/scene_file")
        self.exec_string = vrep_exec+scene+" &"

        self.debug = kwargs['debug']
        self.addr = kwargs['server_addr']
        self.port = kwargs['server_port']

        vrep_env.VrepEnv.__init__(
            self, self.addr, self.port)

        ### ------------  ROS INITIALIZATION  ------------ ###
        self.rate = kwargs['rate']
        # Subscribers / Publishers / TimerCallback
        self.key_sub = rospy.Subscriber(
            "key_input", Int8, self._keys, queue_size=10)
        self.rs_image_sub = rospy.Subscriber(
            "/vrep/depth_image", Image, self._depth_CB, queue_size=1)
        self.pressure_sub = rospy.Subscriber(
            "/vrep/pressure_data", Float32MultiArray, self._pressure_CB, queue_size=1)
        self.reset_pub = rospy.Publisher("reset_key", Int8, queue_size=1)
        self.quit_pub = rospy.Publisher("quit_key", Int8, queue_size=1)
        self.key_pub = rospy.Publisher(
            "rl_key_output", Float32MultiArray, queue_size=1)
        self.jointPub_ = rospy.Publisher(
            "j2n6s300/joint_states", JointState, queue_size=1)
        self.feedbackPub_ = rospy.Publisher(
            "feedback_states", FollowJointTrajectoryFeedback, queue_size=1)
        self.publishWorkerTimer_ = rospy.Timer(
            kwargs['period'], self._publishWorker)
        self.worker_pause = False

        ### ------------  ACTION LIBRARY INITIALIZATION  ------------ ###
        self._action_name = "j2n6s300/follow_joint_trajectory"
        self.trajAS_ = SimpleActionServer_mod(
            self._action_name, FollowJointTrajectoryAction, self._trajCB, False)
        self.trajAS_.start()

        ### ------------  JOINT HANDLES INITIALIZATION  ------------ ###
        self.jointState_ = JointState()
        self.feedback_ = FollowJointTrajectoryFeedback()
        self.jointHandles_ = []
        # Joint prefix setup
        vrepArmPrefix = "jaco_joint_"
        vrepFingerPrefix = "jaco_joint_finger_"
        vrepFingerTipPrefix = "jaco_joint_finger_tip_"
        urdfArmPrefix = "j2n6s300_joint_"
        urdfFingerPrefix = "j2n6s300_joint_finger_"
        urdfFingerTipPrefix = "j2n6s300_joint_finger_tip_"
        # Handle init
        self.jointHandles_ = self._initJoints(
            vrepArmPrefix, vrepFingerPrefix, vrepFingerTipPrefix,
            urdfArmPrefix, urdfFingerPrefix, urdfFingerTipPrefix)
        # Feedback message initialization
        for i in range(0, 6):
            self.feedback_.joint_names.append(self.jointState_.name[i])
            self.feedback_.actual.positions.append(0)
        self.gripper_angle_1 = 0.35    # finger 1, 2
        self.gripper_angle_2 = 0.35    # finger 3
        self.gripper_angle = 0.35      # finger angle of manual control

        ### ------------  STATE GENERATION  ------------ ###
        self.state_gen = kwargs['stateGen']
        self.image_buffersize = 5
        self.image_buff = []
        self.pressure_buffersize = 100
        self.pressure_state = []

        self.depth_trigger = True
        self.pressure_trigger = True
        self.data_buff = []
        self.data_buff_temp = [0, 0, 0]

        ### ------------  REWARD  ------------ ###
        self.reward_method = kwargs['reward_method']
        self.reward_module = kwargs['reward_module']

    def _publishWorker(self, e):
        if not self.worker_pause:
            try:
                self._updateJointState()
                self._publishJointInfo()
            except Exception:
                pass

    def _updateJointState(self):
        self.jointState_.header.stamp = rospy.Time.now()
        position = []
        for i_jointhandle in self.jointHandles_:
            position.append(self.obj_get_joint_angle(i_jointhandle))
        self.jointState_.position = position

    def _publishJointInfo(self):
        self.jointPub_.publish(self.jointState_)
        self.feedback_.header.stamp = rospy.Time.now()
        self.feedback_.actual.positions = self.jointState_.position
        self.feedbackPub_.publish(self.feedback_)

    def _trajCB(self, goal):
        result = FollowJointTrajectoryResult()
        points = goal.trajectory.points
        startTime = rospy.Time.now()
        position = []
        try:
            for i_jointhandle in self.jointHandles_:
                position.append(self.obj_get_joint_angle(i_jointhandle))
            print("Position_now: ",position[:6])
            self.jointState_.position = position
            startPos = self.jointState_.position
            i = len(points)-2
            print("Position_goal: ",points[i].positions[:6])
            move_diff = np.linalg.norm(np.array(points[i].positions[:6])-np.array(position[:6]))
            if (not move_diff > 1) or (not move_diff < 6):
                while not rospy.is_shutdown():
                    if self.trajAS_.is_preempt_requested():
                        self.trajAS_.set_preempted()
                        break
                    fromStart = rospy.Time.now() - startTime
                    while i < len(points) - 1 and points[i+1].time_from_start < fromStart:
                        i += 1
                    if i == len(points)-1:
                        self.reachedGoal = True
                        for j in range(6):
                            tolerance = 0.1
                            if len(goal.goal_tolerance) > 0:
                                tolerance = goal.goal_tolerance[j].position
                            if abs(self.jointState_.position[j] - points[i].positions[j]) > tolerance:
                                self.reachedGoal = False
                                break
                        timeTolerance = rospy.Duration(
                            max(goal.goal_time_tolerance.to_sec(), 0.1))
                        if self.reachedGoal:
                            result.error_code = result.SUCCESSFUL
                            self.trajAS_.set_succeeded(result)
                            break
                        elif fromStart > points[i].time_from_start + timeTolerance:
                            result.error_code = result.GOAL_TOLERANCE_VIOLATED
                            self.trajAS_.set_aborted(result)
                            break
                        target = points[i].positions
                    else:
                        fromStart = rospy.Time.now() - startTime
                        timeTolerance = rospy.Duration(
                            max(goal.goal_time_tolerance.to_sec(), 0.7))
                        if fromStart > points[i].time_from_start + timeTolerance or fromStart < rospy.Duration(0):
                            result.error_code = result.GOAL_TOLERANCE_VIOLATED
                            self.trajAS_.set_aborted(result)
                            break
                        try:
                            if i == 0:
                                segmentDuration = points[i].time_from_start
                                prev = startPos
                            else:
                                segmentDuration = points[i].time_from_start - \
                                    points[i-1].time_from_start
                                prev = points[i-1].positions
                            if segmentDuration.to_sec() <= 0:
                                target = points[i].positions
                            else:
                                #d = fromStart - points[i].time_from_start
                                #alpha = d.to_sec() / segmentDuration.to_sec()
                                # target = self._interpolate(
                                #    prev, points[i].positions, alpha)
                                target = points[i].positions
                        except Exception as e:
                            target = [0, 0, 0, 0, 0, 0]
                            print("Error: ", e)
                    for j in range(0, 6):
                        self.obj_set_position_target(
                            self.jointHandles_[j], radtoangle(-target[j]))
                    self.rate.sleep()
            else:
                result.error_code = result.GOAL_TOLERANCE_VIOLATED
                self.trajAS_.set_aborted(result)
        except Exception as e:
            print(e, file=sys.stderr)
            

    def _interpolate(self, last, current, alpha):
        intermediate = []
        for i in range(0, len(last)):
            intermediate.append(last[i] + alpha * (current[i] - last[i]))
        return intermediate

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

    def _keys(self, msg):
        self.key_input = msg.data
        #print("input = ", self.key_input)
        if self.key_input == ord('r'):      # Reset environment
            self._reset()
            self.key_input = ord('1')
        elif self.key_input == ord('t'):    # Reset environment (step-wised)
            self._reset(True)
        elif self.key_input == ord('n'):    # Next step
            self.step_simulation()
        elif self.key_input == ord('p'):    # Action from Policy
            self.action_from_policy = True
        elif self.key_input == ord('s'):    # Action from Sample
            self.action_from_policy = False
        elif self.key_input in [ord('o'), ord('c')]:
            self._take_manual_action(self.key_input)

    def _reset(self, sync=False):
        self.reset_pub.publish(Int8(data=ord('r')))
        time.sleep(1.2)
        self.gripper_angle_1 = 0
        self.gripper_angle_2 = 0
        self.gripper_angle = 0
        self.trajAS_.reset()

        proc_reset = self._memory_check()
        self._vrep_process_reset() if proc_reset else None

        if self.sim_running:
            self.stop_simulation()

        random_init_angle = [sample(range(-180, 180), 1)[0], 150, sample(range(200, 270), 1)[0], sample(
            range(50, 130), 1)[0], sample(range(50, 130), 1)[0], sample(range(50, 130), 1)[0]]  # angle in degree
        for i, degree in enumerate(random_init_angle):
            noise = randint(-20, 20)
            self.obj_set_position_inst(self.jointHandles_[i], -degree+noise)
            self.obj_set_position_target(self.jointHandles_[i], -degree+noise)
        self.start_simulation(sync=sync, time_step=0.05)
        if sync:
            self.step_simulation()
            time.sleep(0.5)
        return self._get_observation()
    
    def _memory_check(self):
        total = psutil.virtual_memory().total
        used = total - psutil.virtual_memory().available
        '''
        for proc in psutil.process_iter():
            try:
                processName = proc.name()
                if processName in ['coppeliaSim','vrep']:
                    print("vrep found")
            except(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        '''
        return True if (used/total*100)>75 else False

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

    def _get_observation(self,target=None):
        # TODO: Use multiprocessing to generate state from parallel computation
        test = True                             #TODO: Remove test
        if test:
            observation = self.obj_get_position(
                self.jointHandles_[5]) + self.obj_get_orientation(self.jointHandles_[5])
            if target == None:
                observation += [0] * 3
            else:
                observation += target
            if np.isnan(np.sum(observation)):
                #print("NAN OCCURED IN VREP CLIENT!!!!!")
                # If nan, try to get observation recursively untill we do not have any nan
                observation = self._get_observation(target)
        else:
            data_from_callback = []
            observation = self.state_gen.generate(data_from_callback)
        return np.array(observation)

    def _get_reward(self, target_pose):
        # TODO: Reward from IRL
        test = True
        if test:                                #TODO: Remove test
            gripper_pose = self._get_observation(target_pose)
        else:
            gripper_pose = self._get_observation()
        if self.reward_method == "l2":
            dist_diff = np.linalg.norm(
                np.array(gripper_pose[:3]) - np.array(target_pose[:3]))
            reward = (3 - dist_diff*1.3)            #TODO: Shape reward
            return reward - 1
        elif self.reward_method == "":
            return self.reward_module(gripper_pose, target_pose)
        else:
            print("Constant Reward. SHOULD BE FIXED")
            return 30
            #raise NameError("Wrong reward type")

    def _get_terminal_inspection(self, target_pose):
        gripper_pose = self._get_observation()
        dist_diff = np.linalg.norm(
            np.array(gripper_pose[:3]) - np.array(target_pose[:3]))
        if dist_diff < 0.2:                     #TODO: Shape terminal inspection
            return True, 10
        else:
            return False, 0

    def _take_action(self, a):
        key_out = Float32MultiArray()  # a = [-1,0,1] * 8
        key_out.data = np.array(a[:6], dtype=np.float32)
        self.key_pub.publish(key_out)
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
        if key == ord("o"):
            inc = 0.05
        elif key == ord("c"):
            inc = -0.05
        else:
            pass
        self.gripper_angle = max(min(self.gripper_angle+inc, 0), -0.7)
        for i in range(6, 9):
            self.obj_set_position_target(
                self.jointHandles_[i], radtoangle(self.gripper_angle))

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
        '''
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1,1,1)
        plt.axis('off')
        # fig.savefig('/home/ljh/Documents/Figure_1.png', bbox_inches='tight',pad_inches=0)
        plt.imshow(data)
        plt.show()'''
        #print("depth image: ", msg_time)
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
