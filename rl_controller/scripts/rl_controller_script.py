#!/usr/bin/env python3

from algo.trpo import TRPO
from algo.trpotrainer import TRPOTrainer
from env.env_real import Real
from env.env_vrep_api import JacoVrepEnv as JacoVrepEnvApi
from env.env_vrep_pyrep import JacoVrepEnv as JacoVrepEnvPyrep
from state_gen.state_generator import State_generator
from std_msgs.msg import Int8
from rl_controller.srv import InitTraining
from argparser import ArgParser
import rospy
from matplotlib import pyplot as plt
from keras import backend as K
import tensorflow as tf
import random
import time
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RL_controller:
    def __init__(self, feedbackRate_=50):
        rospy.init_node("RL_controller", anonymous=True)
        self.use_sim = rospy.get_param("/rl_controller/use_sim")
        self.trainig_srv = rospy.Service('policy_train', InitTraining, self._train)
        self.learningkey_pub = rospy.Publisher("learning_key", Int8, queue_size=1)

        # Arguments
        parser = ArgParser()
        args = parser.parse_args()

        # Debug
        args.debug = True
        print("DEBUG = ", args.debug)

        # TensorFlow Setting
        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
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
        self.env = JacoVrepEnvApi(
            **vars(args)) if self.use_sim else Real(**vars(args))
        args.env = self.env

        # If resume training on pre-trained models with episodes, else None
        args.model_path = "/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/models_jointpose/"
        os.makedirs(args.model_path, exist_ok=True)
        #args.training_index = 192
        self.trainer = TRPOTrainer(**vars(args))

    def _train(self, req):
        print("Training service init")
        with self.sess as sess:
            learning_key = Int8()
            learning_key.data = 1
            self.learningkey_pub.publish(learning_key)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            K.set_session(sess)
            self.trainer.train(session=sess)
            print("Train Finished")


if __name__ == "__main__":
    try:
        controller_class = RL_controller()
        rospy.spin()
        if controller_class.use_sim:
            print(controller_class.env.close)
            controller_class.env.close()
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")
        controller_class.sess.close()
