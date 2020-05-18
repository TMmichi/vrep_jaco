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
        self.debug = kwargs['debug']
        kwargs['server_addr']=server_addr
        kwargs['server_port']=server_port
        super().__init__(**kwargs)

        ### ------------  RL SETUP  ------------ ###
        self.current_steps = 0
        self.num_envs = 1

        try:
            self.state_shape = kwargs['stateGen'].get_state_shape()
        except Exception:
            self.state_shape = [9]
        self.obs_max = 2
        obs = np.array([self.obs_max]*self.state_shape[0])
        self.observation_space = spaces.Box(-obs, obs)

        self.action_space_max = 1    				# 0.01 (m/s)
        act = np.array([self.action_space_max]*8) 	# x,y,z,r,p,y, finger 1/2, finger 3
        self.action_space = spaces.Box(-act, act)	# Action space: [-0.01, 0.01]
        
        self.seed()
        self.reset()


    def reset(self, sync=False):
        self.current_steps = 0
        return self._reset()

    def get_state_shape(self):
        return self.state_shape

    def get_num_action(self):
        return self.action_space.shape[0]

    def get_action_bound(self):
        return self.action_space_max

    def step(self, action):
        # TODO: Determine how many time steps should be proceed when called
        num_step_pass = 1
        # actions = np.clip(actions,-self.action_space_max, self.action_space_max)
        assert self.action_space.contains(
            action), "Action {} ({}) is invalid".format(action, type(action))
        self.take_action(action)
        for _ in range(num_step_pass):
            # TODO: wait for step signal
            self.step_simulation()
        self.make_observation()
        reward_val = self._get_reward()
        done, additional_reward = self.terminal_inspection()
        return self.observation, reward_val + additional_reward, done, {0:0}

    def terminal_inspection(self):
        # TODO: terminal state definition
        test = False
        temp_max_step = 1000
        if test:
            self.current_steps += 1
            return False, 0 if self.current_steps < 32 else True, 0
        else:
            self.current_steps += 1
            if self.current_steps < temp_max_step:
                return self._get_terminal_inspection()
            else:
                return True, 0

    def make_observation(self):
        self.observation = self._get_observation()
        assert self.state_shape[0] == self.observation.shape[0], \
            "State shape from state generator and observations differs. Possible test code error. You should fix it."

    def take_action(self, a):
        self._take_action(a)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
