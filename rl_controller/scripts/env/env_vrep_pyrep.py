#!/usr/bin/env python

import os
import numpy as np
import datetime

from gym import spaces
from gym.utils import seeding

from env.env_vrep_pyrep_util import JacoVrepEnvUtil, radtoangle


class JacoVrepEnv(JacoVrepEnvUtil):
    def __init__(self, **kwargs):
        self.debug = kwargs['debug']
        super().__init__(**kwargs)

        ### ------------  RL SETUP  ------------ ###
        self.current_steps = 0
        self.num_envs = 1
        try:
            self.max_steps = kwargs['steps_per_batch'] * \
                kwargs['batches_per_episodes']
        except Exception:
            self.max_steps = 500
        try:
            self.state_shape = kwargs['stateGen'].get_state_shape()
        except Exception:
            self.state_shape = [9]
        self.obs_max = 2
        obs = np.array([self.obs_max]*self.state_shape[0])
        self.observation_space = spaces.Box(-obs, obs)
        # 0.007 (m) -> multiplied by factor of 2, will later be divided into 2 @ step
        self.action_space_max = 0.7 * 2
        # unit action (1) from the policy = 0.5 (cm) in real world
        # x,y,z,r,p,y, finger {1,2}, finger 3
        act = np.array([self.action_space_max]*8)
        self.action_space = spaces.Box(-act, act)  # Action space: [-1.4, 1.4]
        self.target = [0,0,0,0,0,0]
        self.seed()
        self.reset()

        ### ------------  LOGGING  ------------ ###
        log_dir ="/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/rl_controller/logs"
        os.makedirs(log_dir, exist_ok = True)
        self.joint_angle_log = open(log_dir+"/log.txt",'w')

    def reset(self, sync=False):
        self.current_steps = 0
        sync = True
        return self._reset(sync=sync)

    def get_state_shape(self):
        return self.state_shape

    def get_num_action(self):
        return self.action_space.shape[0]

    def get_action_bound(self):
        return self.action_space_max

    def step(self, action):
        #then = datetime.datetime.now()
        #print("Within the step at: ",then)
        '''
            if not self.trajAS_.execute_thread.is_alive():
                print("Thread Dead")
            else:
                print( "Thread alive")
        '''
        # TODO: Determine how many time steps should be proceed when called
        # moveit trajectory planning (0.15) + target angle following (0.3 - 0.15?)
        # -> In real world, full timesteps are used for conducting action (No need for finding IK solution)
        num_step_pass = 2
        # actions = np.clip(actions,-self.action _space_max, self.action_space_max)
        assert self.action_space.contains(
            action), "Action {} ({}) is invalid".format(action, type(action))
        self.take_action(action/2)
        for _ in range(num_step_pass):
            self.step_simulation()
        #print("Sim stepping takes: ",datetime.datetime.now() - then)
        self.make_observation()
        #print("Making observation takes: ",datetime.datetime.now() - then)
        reward_val = self._get_reward()
        #print("Receiving rew takes: ",datetime.datetime.now() - then)
        try:
            #write_str = "Target: {0:.3f}, {1:.3f}, {2:.3f}, {3:.3f} {4:.3f}, {5:.3f} | Obs: {6:.3f}, {7:.3f}, {8:.3f}, {9:.3f}, {10:.3f}, {11:.3f} | {12:.3f}, {13:.3f}, {14:.3f}, {15:.3f}, {16:.3f}, {17:.3f}, {18:.3f}, {19:.3f}, {20:.3f} | \033[92m Reward: {21:.5f}\033[0m".format(
            #    self.target[0], self.target[1], self.target[2], self.target[3], self.target[4], self.target[5], self.obs[0], self.obs[1], self.obs[2], self.obs[3], self.obs[4], self.obs[5], self.obs[6], self.obs[7], self.obs[8], self.obs[9], self.obs[10], self.obs[11], self.obs[12], self.obs[13], self.obs[14], reward_val)
            write_str = "Target: {0:.3f}, {1:.3f}, {2:.3f}, {3:.3f} {4:.3f}, {5:.3f} | Obs: {6:.3f}, {7:.3f}, {8:.3f}, {9:.3f}, {10:.3f}, {11:.3f} | \033[92m Reward: {21:.5f}\033[0m".format(
                self.target[0], self.target[1], self.target[2], self.target[3], self.target[4], self.target[5], self.obs[0], self.obs[1], self.obs[2], self.obs[3], self.obs[4], self.obs[5], self.obs[6], self.obs[7], self.obs[8], reward_val)
            print(write_str, end='\r')
            self.joint_angle_log.writelines(write_str+"\n")
        except Exception:
            pass
        #print("Printing takes: ",datetime.datetime.now() - then)
        done, additional_reward = self.terminal_inspection()
        #print("\033[31mWhole step takes: ",datetime.datetime.now() - then,"\033[0m")
        return self.obs, reward_val + additional_reward, done, {0: 0}

    def terminal_inspection(self):
        # TODO: terminal state definition
        test = False
        if test:
            self.current_steps += 1
            return False, 0 if self.current_steps < 32 else True, 0
        else:
            self.current_steps += 1
            if self.current_steps < self.max_steps:
                return self._get_terminal_inspection()
            else:
                return True, 0

    def make_observation(self):
        self.obs, self.target = self._get_observation()
        assert self.state_shape[0] == self.obs.shape[0], \
            "State shape from state generator and observations differs. Possible test code error. You should fix it."

    def take_action(self, a):
        #print("action send to moveit at: ",datetime.datetime.now())
        then = datetime.datetime.now()
        self.action_received = False
        self._take_action(a)
        #print("Before loop: ",datetime.datetime.now())
        while not self.action_received:
            if (datetime.datetime.now() - then).total_seconds() > 0.25: #Should be a bit greater than then moveit timeout, ELSE ERROR!
                #print("\033[31mPlan not found\033[0m")
                break
            pass
        #print("action done at: ",datetime.datetime.now())
        self.action_received = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
