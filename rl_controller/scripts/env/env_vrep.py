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
        self.action_space_max = 3    				# 0.01 (m/s)
        act = np.array([self.action_space_max]*8) 	# x,y,z,r,p,y, finger 1/2, finger 3
        self.action_space = spaces.Box(-act, act)	# Action space: [-0.01, 0.01]
        self.state_shape = kwargs['stateGen'].get_state_shape()
        self.seed()
        self.reset_environment()

    def reset_environment(self, sync=False):
        self.current_steps = 0
        return self._reset()

    def get_state_shape(self):
        return self.state_shape

    def get_num_action(self):
        return self.action_space.shape[0]

    def get_action_bound(self):
        return self.action_space_max

    def step(self, action, target_pose=None):
        # TODO: Determine how many time steps should be proceed when called
        num_step_pass = 4
        # actions = np.clip(actions,-self.action_space_max, self.action_space_max)
        assert self.action_space.contains(
            action), "Action {} ({}) is invalid".format(action, type(action))
        
        for _ in range(num_step_pass):
            # TODO: wait for step signal
            self.step_simulation()

        self.make_observation(target_pose)
        
        reward_val = self._reward(target_pose)
        done, additional_reward = self.terminal_inspection(target_pose)
        return self.observation, reward_val + additional_reward, done

    def _reward(self,target_pose):
        return self._get_reward(target_pose)

    def terminal_inspection(self, target_pose):
        # TODO: terminal state definition
        test = False
        temp_max_step = 20
        if test:
            self.current_steps += 1
            return False, 0 if self.current_steps < 32 else True, 0
        else:
            self.current_steps += 1
            if self.current_steps < temp_max_step:
                return self._get_terminal_inspection(target_pose)
            else:
                return True, 0

    def make_observation(self, target_pose=None):
        self.observation = self._get_observation(target_pose)
        assert self.state_shape[0] == self.observation.shape[0], \
            "State shape from state generator and observations differs. Possible test code error. You should fix it."

    def take_action(self, a):
        self._take_action(a)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
