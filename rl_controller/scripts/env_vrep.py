from environment import Environment
import gym
import numpy as np

class Vrep(Environment):

    def __init__(self, **kwargs):
        super().__init__()
        self.env = self.create_environment(**kwargs)
        self.reset_environment()
        self.finished = False

    def create_environment(self, **kwargs):
        #TODO: Create Vrep environment
        env = gym.make(kwargs['environ_string'])
        return env

    def reset_environment(self):
        #TODO: Vrep reset arguemnt
        #Return: Current state from VAE
        self.current_state = self.env.reset()
        return self.current_state

    def get_state_shape(self):
        #TODO: Vrep obs space
        return self.env.observation_space.shape

    def get_num_action(self):
        #TODO: Vrep act space
        return self.env.action_space.shape[0]

    def get_action_bound(self):
        #TODO: Vrep action bound
        return self.env.action_space.high

    def perform_action(self, action):
        #TODO: Vrep action perform -> kinova
        #Return: Next_state (From VAE), reward, terminal, info _ 
        next_state, reward, terminal = self.env.step(action)
        return np.reshape(next_state, [-1,]), reward, terminal

    def set_seed(self, seed):
        #TODO: Vrep seed ??
        self.env.seed(seed)

