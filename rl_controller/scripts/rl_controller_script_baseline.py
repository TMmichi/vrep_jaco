#!/usr/bin/env python3

import os
import math
import rospy
from rl_controller.srv import InitTraining
from std_msgs.msg import Int8

import stable_baselines.common.tf_util as tf_util
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac import SAC
from stable_baselines.sac.policies import MlpPolicy as MlpPolicy_sac, LnMlpPolicy as LnMlpPolicy_sac
from state_gen.state_generator import State_generator
from env.env_vrep_api import JacoVrepEnv as JacoVrepEnvApi
#from env.env_vrep_pyrep import JacoVrepEnv as JacoVrepEnvPyrep
from env.env_real import Real
from argparser import ArgParser


class RL_controller:
    def __init__(self, feedbackRate_=50):
        rospy.init_node("RL_controller", anonymous=True)
        self.use_sim = rospy.get_param("/rl_controller/use_sim")
        self.trainig_srv = rospy.Service(
            'policy_train', InitTraining, self._train)
        self.learningkey_pub = rospy.Publisher(
            "learning_key", Int8, queue_size=1)

        # Arguments
        parser = ArgParser(isbaseline=True)
        args = parser.parse_args()

        # Debug
        args.debug = True
        print("DEBUG = ", args.debug)

        # TensorFlow Setting
        self.sess = tf_util.single_threaded_session()
        args.sess = self.sess

        # State Generation Module defined here
        self.stateGen = State_generator(**vars(args))
        args.stateGen = self.stateGen

        # Reward Generation
        self.reward_method = "l2"
        self.reward_module = ""
        args.reward_method = self.reward_method
        args.reward_module = self.reward_module

        # ROS settings
        self.rate = rospy.Rate(feedbackRate_)
        self.period = rospy.Duration(1.0/feedbackRate_)
        args.rate = self.rate
        args.period = self.period

        # If resume training on pre-trained models with episodes, else None
        self.model_path = "/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/models_baseline/"
        args.model_path = self.model_path
        self.tb_dir = "/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/tensorboard_log"
        args.tb_dir = self.tb_dir

        self.steps_per_batch = 100
        self.batches_per_episodes = 5
        args.steps_per_batch = self.steps_per_batch
        args.batches_per_episodes = self.batches_per_episodes
        self.num_episodes = 1000
        self.train_num = 1
        self.env = JacoVrepEnvApi(
            **vars(args)) if self.use_sim else Real(**vars(args))
        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * \
            math.ceil(self.num_episodes / self.train_num)
        # self.trainer = TRPO(MlpPolicy, self.env, cg_damping=0.1, vf_iters=5, vf_stepsize=1e-3, timesteps_per_batch=self.steps_per_batch,
        #                    tensorboard_log=args.tb_dir, full_tensorboard_log=True)
        self.trainer = SAC(
            LnMlpPolicy_sac, self.env, tensorboard_log=args.tb_dir, full_tensorboard_log=True)
        

    def _train(self, req):
        print("Training service init")
        with self.sess:
            for train_iter in range(self.train_num):
                print("Training Iter: ", train_iter)
                model_dir=self.model_path + str(rospy.Time.now())
                os.makedirs(model_dir, exist_ok = True)
                learning_key=Int8()
                learning_key.data=1
                self.learningkey_pub.publish(learning_key)
                self.trainer.learn(total_timesteps = self.num_timesteps)
                print("Train Finished")
                self.trainer.save(model_dir)


if __name__ == "__main__":
    try:
        controller_class=RL_controller()
        rospy.spin()
        if controller_class.use_sim:
            print(controller_class.env.close)
            controller_class.env.joint_angle_log.close()
            controller_class.env.close()
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")
        controller_class.sess.close()
