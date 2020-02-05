#!/usr/bin/env python

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent

import os
import time
import math
import random
from random import sample

import gym
from gym import spaces
from gym.utils import seeding

import rospy
import actionlib
from actionlib import SimpleActionServer
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

key_dict = {97:"a",98:"b",99:"c",100:"d",101:"e",102:"f",103:"g",104:"h",
105:"i",106:"j",107:"k",108:"l",109:"m",110:"n",111:"o",112:"p",113:"q",
114:"r",115:"s",116:"t",117:"u",118:"v",119:"w",120:"x",121:"y",122:"z"}


def radtoangle(rad):
	return rad / math.pi * 180


class JacoVrepEnv(vrep_env.VrepEnv):
	metadata = {'render.modes': []}
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
		feedbackRate_=50.0):
		
		rospy.init_node("JacoVrepEnv",anonymous=True)
		self.rate = rospy.Rate(50)

		#Key input subscriber
		self.key_sub = rospy.Subscriber("key_input",Int8,self._keys, queue_size=10)
		self.keychk_pub = rospy.Publisher("key_check",Int8,queue_size=10)
		self.key_input = 0
		
		#Initialize Vrep API
		vrep_env.VrepEnv.__init__(self,server_addr,server_port)

		### ------  PREREQUESITES FOR ACTION LIBRARY  ------ ###
		self.jointState_ = JointState()
		self.feedback_ = FollowJointTrajectoryFeedback()
		self.jointHandles_ = []
		self.jointPub_ = rospy.Publisher("jaco/joint_states",JointState,queue_size=1)
		self.feedbackPub_ = rospy.Publisher("feedback_states",FollowJointTrajectoryFeedback,queue_size=1)
		self.publishWorkerTimer_ = rospy.Timer(rospy.Duration(1.0/feedbackRate_), self._publishWorker)

		### ------  ACTION LIBRARY INITIALIZATION  ------ ###
		self._action_name = "jaco/joint_trajectory_action"
		self.trajAS_ = SimpleActionServer(self._action_name, FollowJointTrajectoryAction, self._trajCB, False)
		self.trajAS_.start()

		#Joint prefix
		vrepRespondablePrefix = "jaco_link_"
		vrepArmPrefix = "jaco_joint_"
		vrepFingerPrefix = "jaco_joint_finger_"
		vrepFingerTipPrefix = "jaco_joint_finger_tip_"
		urdfArmPrefix = "jaco_joint_"
		urdfFingerPrefix = "jaco_joint_finger_"
		urdfFingerTipPrefix = "jaco_joint_finger_tip_"

		#Joint initialization
		self.jointHandles_, self.respondHanldes_ = self._initJoints(
			vrepArmPrefix,vrepFingerPrefix,vrepFingerTipPrefix,
			urdfArmPrefix,urdfFingerPrefix,urdfFingerTipPrefix,
			vrepRespondablePrefix)
		
		#Feedback message initialization
		for i in range(0,6):
			self.feedback_.joint_names.append(self.jointState_.name[i])
			self.feedback_.actual.positions.append(0)

		'''
		### ------  STATE GENERATION  ------ ###
		# State Generator
		self.state_gen = State_generator(**kwargs)
		'''

		# #modify: if size of action space is different than number of joints
		# Example: One action per joint
		num_act = 9

		# #modify: action_space and observation_space to suit your needs
		self.joints_max_velocity = 3.0
		act = np.array( [self.joints_max_velocity] * num_act )

		self.action_space      = spaces.Box(-act,act)
		self.seed()
		self.reset()


	def _trajCB(self, goal):
		result = FollowJointTrajectoryResult()
		points = goal.trajectory.points
		startTime = rospy.Time.now()
		startPos = self.jointState_.position
		i = 0
		while not rospy.is_shutdown():
			if self.trajAS_.is_preempt_requested():
				self.trajAS_.set_preempted()
				break
			fromStart = rospy.Time.now() - startTime
			while i < len(points) - 1 and points[i+1].time_from_start < fromStart:
				i += 1
			if i == len(points)-1:
				reachedGoal = True
				for j in range(6):
					tolerance = 0.1
					if len(goal.goal_tolerance) > 0:
						tolerance = goal.goal_tolerance[j].position
					if abs(self.jointState_.position[j] - points[i].positions[j]) > tolerance:
						reachedGoal = False
						break
				timeTolerance = rospy.Duration(max(goal.goal_time_tolerance.to_sec(),0.1))
				if reachedGoal:
					result.error_code = result.SUCCESSFUL
					self.trajAS_.set_succeeded(result)
					break
				elif fromStart > points[i].time_from_start + timeTolerance:
					result.error_code = result.GOAL_TOLERANCE_VIOLATED
					self.trajAS_.set_aborted(result)
					break
				target = points[i].positions
			else:
				if i==0:
					segmentDuration = points[i].time_from_start
					prev = startPos
				else:
					segmentDuration = points[i].time_from_start - points[i-1].time_from_start
					prev = points[i-1].positions
				if segmentDuration.to_sec() <= 0:
					target = points[i].positions
				else:
					d = fromStart - points[i].time_from_start
					alpha = d.to_sec() / segmentDuration.to_sec()
					target = self._interpolate(prev, points[i].positions, alpha)
			print("Ready to be moved")
			try:
				for i in range(0,9):
					self.obj_set_position_target(self.jointHandles_[i],radtoangle(-target[i]))
					print("for looped: ",target[i])
			except Exception:
				for i in range(0,6):
					self.obj_set_position_target(self.jointHandles_[i],radtoangle(-target[i]))
				print("chunck: ",target)
				#map(self.obj_set_position_target,self.jointHandles_,target)
			print("Moving")
			self.rate.sleep()

	def _interpolate(self, last, current, alpha):
		intermediate = []
		for i in range(0,len(last)):
			intermediate.append(last[i] + alpha * (current[i] - last[i]))
		return intermediate

	def _initJoints(
		self,
		vrepArmPrefix,vrepFingerPrefix,vrepFingerTipPrefix,
		urdfArmPrefix,urdfFingerPrefix,urdfFingerTipPrefix,
		vrepRespondablePrefix):
		"""Initialize joints object handles and joint states
		"""
		in_names = []
		resp_names = []
		for i in range(1,7):
			in_names.append(vrepArmPrefix+str(i))
			outname = urdfArmPrefix+str(i)
			self.jointState_.name.append(outname)
			self.jointState_.velocity.append(0)
			self.jointState_.effort.append(0)
			if i != 6:
				resp_names.append(vrepRespondablePrefix+str(i)+"_respondable")
			else:
				resp_names.append(vrepRespondablePrefix+"hand_respondable")
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
		respondHanldes_ = list(map(self.get_object_handle, resp_names))
		self.jointState_.position = list(map(self.obj_get_joint_angle,jointHandles_))
		return jointHandles_, respondHanldes_

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

	def _keys(self,msg):
		self.key_input = msg.data
		self.keychk_pub.publish(self.key_input)
		self.key_input = key_dict[self.key_input]
		print("input = ",self.key_input)
		
		if self.key_input == "r": #Reset environment
			self.reset()
			self.key_input = "1"
		elif self.key_input == "t": #Reset environment (step-wised)
			self.reset(True)
		elif self.key_input == "n": #Next step
			self.step_simulation()
		elif self.key_input == "p": #Action from Policy
			self.action_from_policy = True
		elif self.key_input == "s": #Action frmo Sample
			self.action_from_policy = False

	def _make_observation(self):
		"""Query V-rep to make observation.
		   The observation is stored in self.observation
		"""
		buff = []
		self.observation = []#self.generate(buff)
	
	def _make_action(self, a):
		"""Query V-rep to make action.
		   no return value
		"""
		# #modify
		# example: set a velocity for each joint
		for i_oh, i_a in zip(self.oh_joint, a):
			self.obj_set_velocity(i_oh, i_a)
	
	def step(self, action):
		"""Gym environment 'step'
		"""
		# #modify Either clip the actions outside the space or assert the space contains them
		# actions = np.clip(actions,-self.joints_max_velocity, self.joints_max_velocity)
		assert self.action_space.contains(action), "Action {} ({}) is invalid".format(action, type(action))
		
		# Actuate
		self._make_action(action)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		# Reward
		# #modify the reward computation
		# example: possible variables used in reward
		head_pos_x = self.observation[0] # front/back
		head_pos_y = self.observation[1] # left/right
		head_pos_z = self.observation[2] # up/down
		nrm_action  = np.linalg.norm(action)
		r_regul     = -(nrm_action**2)
		r_alive = 1.0
		# example: different weights in reward
		reward = (8.0)*(r_alive) +(4.0)*(head_pos_x) +(1.0)*(head_pos_z)
		
		# Early stop
		# #modify if the episode should end earlier
		tolerable_threshold = 0.20
		done = (head_pos_z < tolerable_threshold)
		#done = False
		
		return self.observation, reward, done, {}
	
	def reset(self,sync=False):
		"""Gym environment 'reset'
		"""
		if self.sim_running:
			self.stop_simulation()
		else:
			self.start_simulation()
			self.stop_simulation()
		random_init_angle = [sample(range(-180,180),1)[0],110,sample(range(20,130),1)[0],sample(range(50,130),1)[0],sample(range(50,130),1)[0],sample(range(50,130),1)[0]] #angle in degree
		for i, degree in enumerate(random_init_angle):
			noise = random.randint(-30,30)
			self.obj_set_position_inst(self.jointHandles_[i],-degree+noise)
			self.obj_set_position_target(self.jointHandles_[i],-degree+noise)
		self.start_simulation(sync)
		if sync:
			self.step_simulation()
			time.sleep(1)
		else:
			pass
		self._make_observation()
		return self.observation
	
	def render(self, mode='human', close=False):
		pass
	
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	

def main(args):
	"""main function used as test and example.
	   Agent does random actions with 'action_space.sample()'
	"""
	# #modify: the env class name
	env = JacoVrepEnv()
	for i_episode in range(8):
		observation = env.reset()
		total_reward = 0
		for t in range(256):
			action = env.action_space.sample()
			observation, reward, done, _ = env.step(action)
			total_reward += reward
			if done:
				break
		print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
	env.close()
	return 0

if __name__ == '__main__':
	try:
		vrepenv_class = JacoVrepEnv()
		rospy.spin()
	except rospy.ROSInitException:
		rospy.loginfo("node terminated.")
