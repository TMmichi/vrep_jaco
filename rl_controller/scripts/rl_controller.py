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
from env.env_real import Real
from env.env_vrep_client import JacoVrepEnv
from algo.trpotrainer import TRPOTrainer
from algo.trpo import TRPO


class RL_controller:
    def __init__(self, feedbackRate_=50):
        rospy.init_node("RL_controller", anonymous=True)
        self.use_sim = rospy.get_param("/rl_controller/use_sim")
        self.trigger_sub = rospy.Subscriber(
            "key_input", Int8, self.trigger, queue_size=10)

        parser = ArgParser()
        args = parser.parse_args()
        tf.compat.v1.reset_default_graph()

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)
        self.sess = tf.compat.v1.Session(config=config)
        args.sess = self.sess
        self.rate = rospy.Rate(feedbackRate_)
        self.period = rospy.Duration(1.0/feedbackRate_)
        args.rate = self.rate
        args.period = self.period
        self.env = JacoVrepEnv(
            **vars(args)) if self.use_sim else Real(**vars(args))
        args.env = self.env
        self.training = False
        self.trainer = TRPOTrainer(**vars(args))

    def trigger(self, msg):
        if msg.data == ord('1'):
            self.agent()
        elif msg.data == ord('2'):
            if not self.training:
                self.train()
                self.training = True

    def agent(self):
        pass

    def train(self):
        with self.sess as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            K.set_session(sess)
            self.trainer.train(session=sess)


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
