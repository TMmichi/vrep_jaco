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
from std_msgs.msg import Bool
from argparser import ArgParser
from env_vrep import Vrep
from env_real import Real


class RL_controller:
    def __init__(self):
        rospy.init_node("RL_controller",anonymous=True)
        self.use_sim = rospy.get_param("/rl_controller/use_sim")

        parser = ArgParser()
        args = parser.parse_args()
        tf.reset_default_graph()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.compat.v1.Session(config=config)
        args.sess = self.sess
        args.env = Vrep(**vars(args)) if self.use_sim else Real(**vars(args))
        self.local_brain = TRPO(**vars(args))

        self.trainer = TRPOTrainer(**vars(args))
        self.agent_trigger = rospy.Subscriber("/agent_on",Bool,self.agent)
        self.train_trigger = rospy.Subscriber("/train_on",Bool,self.train)

    
    def spin(self, rate):
        self.rate = rospy.Rate(rate)
        rospy.spin()

    def agent(self, msg):
        pass

    def train(self,msg):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        K.set_session(self.sess)
        self.trainer.train(session=self.sess)


if __name__=="__main__":
    try:
        controller_class = RL_controller()
        controller_class.spin(10)
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")
        controller_class.sess.close()