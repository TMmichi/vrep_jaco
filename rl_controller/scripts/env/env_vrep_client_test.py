#!/usr/bin/env python

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent
from env_util_test import JacoVrepEnvUtil, radtoangle

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


class JacoVrepEnv(JacoVrepEnvUtil):
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
        **kwargs):
		super().__init__(**kwargs)
		### ------------  V-REP API INITIALIZATION  ------------ ###
		vrep_env.VrepEnv.__init__(self,server_addr,server_port)		

		### ------------  RL SETUP  ------------ ###
		'''
		# State Generator
		self.state_gen = State_generator(**kwargs)
		'''
		self.action_space_max = 3.0
		act = np.array([self.action_space_max]*8) #x,y,z,r,p,y, finger 1/2, finger 3
		self.action_space = spaces.Box(-act,act)
		self.seed()
		self.reset_environment()


	def reset_environment(self,sync=False):
		return self._reset()

	def get_state_shape(self):
		return self.action_space.shape

	def get_num_action(self):
		return self.action_space.shape[0]

	def get_action_bound(self):
		return self.action_space_max

	def step(self, action):
		# actions = np.clip(actions,-self.action_space_max, self.action_space_max)
		assert self.action_space.contains(action), "Action {} ({}) is invalid".format(action, type(action))
		self.take_action(action)
		self.step_simulation()
		self.make_observation()
		reward = 0
		done = False
		return self.observation, reward, done, {}
	
	def make_observation(self):
		self.observation = self._get_observation()
	
	def take_action(self, a):
		self._take_action(a)
		
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
	