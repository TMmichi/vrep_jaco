#!/usr/bin/env python3

import os
import math
import rospy
import numpy as np
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
        self.use_sim = rospy.get_param("/rl_controller/use_sim")
        rospy.init_node("RL_controller", anonymous=True)
        self.cip_test_srv = rospy.Service(
            'cip_test', InitTraining, self._CIP_test)
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

        self.env = JacoVrepEnvApi(
            **vars(args)) if self.use_sim else Real(**vars(args))


    def _CIP_test(self, req):
        roll = True
        manual = True
        if manual:
            while roll:
                cmd = input("Action/quit (pose[m/rad] / q): ")
                try:
                    cmd = np.array(cmd.split(","),dtype=np.float16)
                    assert len(cmd) == 6, "Input command should be in a size of 6"
                    self.env.take_pose_action(cmd)
                except Exception:
                    if cmd == 'q':
                        break
                    else:
                        print("[ERROR] Input: pose action / q for quit")
        else:
            pass


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
