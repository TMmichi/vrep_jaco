#!/usr/bin/env python

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent

import os
import sys
import time
from math import pi
import random
from random import sample

import gym
from gym import spaces
from gym.utils import seeding

import rospy
import actionlib
#from actionlib import SimpleActionServer
from SimpleActionServer_mod import SimpleActionServer_mod
from actionlib.server_goal_handle import ServerGoalHandle
from moveit_msgs.msg import DisplayTrajectory
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryFeedback
from control_msgs.msg import FollowJointTrajectoryResult
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Pose

import numpy as np


def radtoangle(rad):
	return rad / pi * 180


class JacoVrepEnv(vrep_env.VrepEnv):
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
        **kwargs):

		### ------------  V-REP API INITIALIZATION  ------------ ###
		vrep_env.VrepEnv.__init__(self,server_addr,server_port)

		### ------------  ROS INITIALIZATION  ------------ ###
		#Initialize rospy node
		rospy.init_node("JacoVrepEnv",anonymous=True)
		self.rate = kwargs['rate']
		#Subscribers / Publishers / Callbacks
		self.key_sub = rospy.Subscriber("key_input",Int8,self._keys, queue_size=10)
		self.jointPub_ = rospy.Publisher("j2n6s300/joint_states",JointState,queue_size=1)
		self.feedbackPub_ = rospy.Publisher("feedback_states",FollowJointTrajectoryFeedback,queue_size=1)
		self.publishWorkerTimer_ = rospy.Timer(kwargs['period'], self._publishWorker)
		
		### ------------  ACTION LIBRARY INITIALIZATION  ------------ ###
		self._action_name = "j2n6s300/follow_joint_trajectory"
		self.trajAS_ = SimpleActionServer_mod(self._action_name, FollowJointTrajectoryAction, self._trajCB, False)
		self.trajAS_.start()

		### ------------  JOINT HANDLES INITIALIZATION  ------------ ###
		self.jointState_ = JointState()
		self.feedback_ = FollowJointTrajectoryFeedback()
		self.jointHandles_ = []
		#Joint prefix setup
		vrepArmPrefix = "jaco_joint_"
		vrepFingerPrefix = "jaco_joint_finger_"
		vrepFingerTipPrefix = "jaco_joint_finger_tip_"
		urdfArmPrefix = "j2n6s300_joint_"
		urdfFingerPrefix = "j2n6s300_joint_finger_"
		urdfFingerTipPrefix = "j2n6s300_joint_finger_tip_"
		#Handle init
		self.jointHandles_ = self._initJoints(
			vrepArmPrefix,vrepFingerPrefix,vrepFingerTipPrefix,
			urdfArmPrefix,urdfFingerPrefix,urdfFingerTipPrefix)
		#Feedback message initialization
		for i in range(0,6):
			self.feedback_.joint_names.append(self.jointState_.name[i])
			self.feedback_.actual.positions.append(0)

		### ------------  RL SETUP  ------------ ###
		'''
		# State Generator
		self.state_gen = State_generator(**kwargs)
		'''
		self.action_space_max = 3.0
		act = np.array([self.action_space_max]*9)
		self.action_space = spaces.Box(-act,act)
		self.seed()
		self.reset_environment()


	def _publishWorker(self, e):
		self._updateJointState()
		self._publishJointInfo()

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
		position=[]
		for i_jointhandle in self.jointHandles_:
			position.append(self.obj_get_joint_angle(i_jointhandle))
		self.jointState_.position = position
		startPos = self.jointState_.position
		i = 0
		while not rospy.is_shutdown():
			if self.trajAS_.is_preempt_requested():
				print("goal preempted")
				self.trajAS_.set_preempted()
				break
			fromStart = rospy.Time.now() - startTime
			while i < len(points) - 1 and points[i+1].time_from_start < fromStart:
				i += 1
			if i == len(points)-1:
				print(self.trajAS_,": Reached", i)
				reachedGoal = True
				for j in range(6):
					tolerance = 0.1
					if len(goal.goal_tolerance) > 0:
						tolerance = goal.goal_tolerance[j].position
					if abs(self.jointState_.position[j] - points[i].positions[j]) > tolerance:
						reachedGoal = False
						print("Error")
						break
				timeTolerance = rospy.Duration(max(goal.goal_time_tolerance.to_sec(),0.1))
				if reachedGoal:
					print("succeded")
					result.error_code = result.SUCCESSFUL
					self.trajAS_.set_succeeded(result)
					break
				elif fromStart > points[i].time_from_start + timeTolerance:
					print("aborted")
					result.error_code = result.GOAL_TOLERANCE_VIOLATED
					self.trajAS_.set_aborted(result)
					break
				target = points[i].positions
			else:
				print(self.trajAS_,": working..", i)
				if i==0:
					segmentDuration = points[i].time_from_start
					prev = startPos
					print("a")
				else:
					segmentDuration = points[i].time_from_start - points[i-1].time_from_start
					prev = points[i-1].positions
					print("b")
				if segmentDuration.to_sec() <= 0:
					target = points[i].positions
					print("c")
				else:
					d = fromStart - points[i].time_from_start
					alpha = d.to_sec() / segmentDuration.to_sec()
					target = self._interpolate(prev, points[i].positions, alpha)
					print("d")
			print("Ready to be moved")
			for j in range(0,6):
				self.obj_set_position_target(self.jointHandles_[j],radtoangle(-target[j]))
			self.rate.sleep()

	def _interpolate(self, last, current, alpha):
		intermediate = []
		for i in range(0,len(last)):
			intermediate.append(last[i] + alpha * (current[i] - last[i]))
		return intermediate

	def _initJoints(
		self,
		vrepArmPrefix,vrepFingerPrefix,vrepFingerTipPrefix,
		urdfArmPrefix,urdfFingerPrefix,urdfFingerTipPrefix):
		"""Initialize joints object handles and joint states
		"""
		in_names = []
		for i in range(1,7):
			in_names.append(vrepArmPrefix+str(i))
			outname = urdfArmPrefix+str(i)
			self.jointState_.name.append(outname)
			self.jointState_.velocity.append(0)
			self.jointState_.effort.append(0)
		for i in range(1,4):
			in_names.append(vrepFingerPrefix+str(i))
			outname = urdfFingerPrefix+str(i)
			self.jointState_.name.append(outname)
			self.jointState_.velocity.append(0)
			self.jointState_.effort.append(0)
		for i in range(1,4):
			in_names.append(vrepFingerTipPrefix+str(i))
			outname = urdfFingerTipPrefix+str(i)
			self.jointState_.name.append(outname)
			self.jointState_.velocity.append(0)
			self.jointState_.effort.append(0)
		jointHandles_ = list(map(self.get_object_handle, in_names))
		self.jointState_.position = list(map(self.obj_get_joint_angle,jointHandles_))
		return jointHandles_

	def _keys(self,msg):
		self.key_input = msg.data
		print("input = ",self.key_input)
		
		if self.key_input == ord('r'): #Reset environment
			self.reset_environment()
			self.key_input = ord(1)
		elif self.key_input == ord('t'): #Reset environment (step-wised)
			self.reset_environment(True)
		elif self.key_input == ord('n'): #Next step
			self.step_simulation()
		elif self.key_input == ord('p'): #Action from Policy
			self.action_from_policy = True
		elif self.key_input == ord('s'): #Action from Sample
			self.action_from_policy = False
		elif self.key_input in [ord('o'),ord('c')]:
			self.take_manual_action(self.key_input)


	def create_environment(self):
		pass

	def reset_environment(self,sync=False):
		"""Gym environment 'reset'
		"""
		print("RESETED")
		self.trajAS_.reset()
		if self.sim_running:
			self.stop_simulation()
		else:
			self.start_simulation(time_step=0.05)
			self.stop_simulation()
		random_init_angle = [sample(range(-180,180),1)[0],150,sample(range(200,270),1)[0],sample(range(50,130),1)[0],sample(range(50,130),1)[0],sample(range(50,130),1)[0]] #angle in degree
		for i, degree in enumerate(random_init_angle):
			noise = random.randint(-20,20)
			self.obj_set_position_inst(self.jointHandles_[i],-degree+noise)
			self.obj_set_position_target(self.jointHandles_[i],-degree+noise)
		self.start_simulation(sync=sync,time_step=0.05)
		if sync:
			self.step_simulation()
			time.sleep(1)
		else:
			pass
		self._make_observation()
		return self.observation

	def get_state_shape(self):
		return self.action_space.shape

	def get_num_action(self):
        return self.action_space.shape[0]

    def get_action_bound(self):
        return self.action_space_max

	def step(self, action):
		"""Gym environment 'step'
		"""
		# #modify Either clip the actions outside the space or assert the space contains them
		# actions = np.clip(actions,-self.action_space_max, self.action_space_max)
		assert self.action_space.contains(action), "Action {} ({}) is invalid".format(action, type(action))
		
		# Actuate
		self._take_action(action)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		# Reward
		reward = 0
		
		# Early stop
		tolerable_threshold = 0.20
		#done = (head_pos_z < tolerable_threshold)
		done = False
		return self.observation, reward, done, {}
	
	def _make_observation(self):
		"""Query V-rep to make observation.
		   The observation is stored in self.observation
		"""
		self.observation = []#self.generate(buff)
	
	def _take_action(self, a):
		pass
	
	def take_manual_action(self,key):
		self.gripper_angle = 0
		if key == "o":
			self.gripper_angle += 1
			for i in range(6,9):
				if self.gripper_angle > 10:
					self.gripper_angle = 10
				self.obj_set_position_target(self.jointHandles_[i],radtoangle(self.gripper_angle))
		elif key == "c":
			self.gripper_angle -= 1
			for i in range(6,9):
				if self.gripper_angle < -20:
					self.gripper_angle = -20
				self.obj_set_position_target(self.jointHandles_[i],radtoangle(-20))
		else:
			pass
	




	
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	