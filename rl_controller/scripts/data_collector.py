#!/usr/bin/env python3

import os
import rospy
from random import randint, sample
from argparser import ArgParser
from std_msgs.msg import Int8
from env.env_vrep import JacoVrepEnv
from env.env_real import Real
import numpy as np


class Data_collector:
    def __init__(self, feedbackRate_=50):
        rospy.init_node("Data_collector", anonymous=True)
        self.use_sim = rospy.get_param("/rl_controller/use_sim")
        self.trigger_sub = rospy.Subscriber(
            "key_input", Int8, self.trigger, queue_size=10)

        #Arguments
        parser = ArgParser()
        args = parser.parse_args()
        args.debug = True
        print("DEBUG = ", args.debug)

        #ROS settings
        self.rate = rospy.Rate(feedbackRate_)
        self.period = rospy.Duration(1.0/feedbackRate_)
        args.rate = self.rate
        args.period = self.period
        self.env = JacoVrepEnv(
            **vars(args)) if self.use_sim else Real(**vars(args))
        args.env = self.env
        args.data_path = "/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/data_stategen/"
        os.makedirs(args.data_path,exist_ok=True)

    def trigger(self, msg):
        if msg.data == ord('1'):
            self._collect()
    
    def _collect(self):
        gripper_pose_list = []
        joint_angle_list = []
        for i in range(20):
            proc_reset = self.env._memory_check()
            self.env._vrep_process_reset() if proc_reset else None
            if self.env.sim_running:
                self.env.stop_simulation()
            sample_angle = [sample(range(-180, 180), 1)[0], 150, sample(range(200, 270), 1)[0], sample(
                range(50, 130), 1)[0], sample(range(50, 130), 1)[0], sample(range(50, 130), 1)[0]]
            joint_angle = []
            for i, degree in enumerate(sample_angle):
                noise = randint(-20, 20)
                self.env.obj_set_position_inst(self.env.jointHandles_[i], -degree+noise)
                self.env.obj_set_position_target(self.env.jointHandles_[i], -degree+noise)
                joint_angle.append(-degree+noise)
            joint_angle_list.append(joint_angle_list)
            self.env.start_simulation(sync=True, time_step=0.05)
            gripper_pose_list.append(self.env.obj_get_position(
                self.env.jointHandles_[5]) + self.env.obj_get_orientation(self.env.jointHandles_[5]))
        for k in range(19):
            for j in range(k+1,20):
                proc_reset = self.env._memory_check()
                self.env._vrep_process_reset() if proc_reset else None
                if self.env.sim_running:
                    self.env.stop_simulation()
                for i, degree in enumerate(gripper_pose_list[k]):
                    self.env.obj_set_position_inst(self.env.jointHandles_[i], joint_angle_list[k][i])
                    self.env.obj_set_position_target(self.env.jointHandles_[i], joint_angle_list[k][i])
                self.env.start_simulation(sync=True, time_step=0.05)
                for i, degree in enumerate(gripper_pose_list[j]):
                    self.env.obj_set_position_inst(self.env.jointHandles_[i], joint_angle_list[j][i])
                    self.env.obj_set_position_target(self.env.jointHandles_[i], joint_angle_list[j][i])
                position = []
                while True:
                    for i_jointhandle in self.env.jointHandles_:
                        position.append(self.env.obj_get_joint_angle(i_jointhandle))
                    self.env.step_simulation()
                    if np.linalg.norm(np.array(position) - np.array(joint_angle_list[j])) < 0.001:
                        break


if __name__ == "__main__":
    try:
        collector_class = Data_collector()
        rospy.spin()
        if collector_class.use_sim:
            print(collector_class.env.close)
            collector_class.env.close()
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")
        collector_class.sess.close()
