#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import time
import random

import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt

import rospy
from argparser import ArgParser
from std_msgs.msg import Int8
from state_gen.state_generator import State_generator
from env.env_vrep import JacoVrepEnv
from env.env_real import Real
from algo.trpotrainer import TRPOTrainer
from algo.trpo import TRPO


class RL_controller:
    def __init__(self, feedbackRate_=50):
        rospy.init_node("RL_controller", anonymous=True)
        self.use_sim = rospy.get_param("/rl_controller/use_sim")
        self.trigger_sub = rospy.Subscriber(
            "key_input", Int8, self.trigger, queue_size=1)

        #Arguments
        parser = ArgParser()
        args = parser.parse_args()

        #Debug
        args.debug = True
        print("DEBUG = ", args.debug)

        #TensorFlow Setting
        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        args.sess = self.sess

        #State Generation Module defined here
        self.stateGen = State_generator(**vars(args))
        args.stateGen = self.stateGen

        #Reward Generation
        self.reward_method = "l2"
        self.reward_module = ""
        args.reward_method = self.reward_method
        args.reward_module = self.reward_module

        #ROS settings
        self.rate = rospy.Rate(feedbackRate_)
        self.period = rospy.Duration(1.0/feedbackRate_)
        args.rate = self.rate
        args.period = self.period
        self.env = JacoVrepEnv(
            **vars(args)) if self.use_sim else Real(**vars(args))
        args.env = self.env

        #Training session
        self.trainingTrigger = False
        #If resume training on pre-trained models with episodes, else None
        args.model_path = "/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/models_jointpose/"
        os.makedirs(args.model_path,exist_ok=True)
        args.training_index = 192
        self.trainer = TRPOTrainer(**vars(args))


    def trigger(self, msg):
        if msg.data == ord('1'):
            self._agent()
        elif msg.data == ord('2'):
            if not self.trainingTrigger:
                self._train()
                self.trainingTrigger = True

    def _agent(self):
        with self.sess as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            K.set_session(sess)

    def _train(self):
        with self.sess as sess:
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
