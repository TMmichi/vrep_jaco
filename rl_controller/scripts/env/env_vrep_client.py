#!/usr/bin/env python

from env.env_vrep_util import JacoVrepEnvUtil, radtoangle

from gym import spaces
from gym.utils import seeding

import numpy as np


class JacoVrepEnv(JacoVrepEnvUtil):
    def __init__(
        self,
        server_addr='127.0.0.1',
        server_port=19997,
            **kwargs):
        kwargs['server_addr']=server_addr
        kwargs['server_port']=server_port
        super().__init__(**kwargs)

        ### ------------  RL SETUP  ------------ ###
        self.current_steps = 0
        self.action_space_max = 3    				# 0.01 (m/s)
        act = np.array([self.action_space_max]*8) 	# x,y,z,r,p,y, finger 1/2, finger 3
        self.action_space = spaces.Box(-act, act)	# Action space: [-0.01, 0.01]
        self.seed()
        self.reset_environment()

    def reset_environment(self, sync=False):
        self.current_steps = 0
        return self._reset()

    def get_state_shape(self):
        return self.action_space.shape

    def get_num_action(self):
        return self.action_space.shape[0]

    def get_action_bound(self):
        return self.action_space_max

    def step(self, action):
        # TODO: Determine how many time steps should be proceed when called
        num_step_pass = 14
        # actions = np.clip(actions,-self.action_space_max, self.action_space_max)
        assert self.action_space.contains(
            action), "Action {} ({}) is invalid".format(action, type(action))
        self.take_action(action)
        for _ in range(num_step_pass):
            # TODO: wait for step signal
            self.step_simulation()
        self.make_observation()
        reward_val = self._reward()
        done = self.terminal_inspection()
        return self.observation, reward_val, done

    def _reward(self):
        # TODO: Reward from IRL
        return 30

    def terminal_inspection(self):
        # TODO: terminal state definition
        self.current_steps += 1
        return False if self.current_steps < 10 else True 

    def make_observation(self):
        self.observation = self._get_observation()

    def take_action(self, a):
        self._take_action(a)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
