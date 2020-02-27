#!/usr/bin/env python3

import time
import random
import math
import cv2 as cv
import numpy as np
import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt

from algo.trpo import TRPO
from algo.trpotrainer import TRPOTrainer

import rospy
from std_msgs.msg import Int8
from argparser import ArgParser
from env_vrep_client_test import JacoVrepEnv
from env_real import Real


class RL_controller:
    def __init__(self, feedbackRate_=50):
        rospy.init_node("RL_controller", anonymous=True)
        self.use_sim = rospy.get_param("/rl_controller/use_sim")
        self.trigger_sub = rospy.Subscriber(
            "key_input", Int8, self.trigger, queue_size=10)

        parser = ArgParser()
        args = parser.parse_args()
        tf.reset_default_graph()

        config = tf.ConfigProto(allow_soft_placement=True,
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

        self.local_brain = TRPO(**vars(args))
        self.trainer = TRPOTrainer(**vars(args))

    def trigger(self, msg):
        if msg.data == ord('1'):
            self.agent()
        elif msg.data == ord('2'):
            self.train()

    def agent(self):
        pass

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        K.set_session(self.sess)
        self.trainer.train(session=self.sess)


if __name__ == "__main__":
    try:
        controller_class = RL_controller()
        rospy.spin()
        controller_class.env.close()
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")
        controller_class.sess.close()
