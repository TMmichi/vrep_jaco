#!/usr/bin/env python3

import time
import math
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt
import numpy as np
#import vae_util

import rospy
from std_msgs.msg import Header
from std_msgs.msg import UInt8
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose


class RL_controller:
    def __init__(self):
        rospy.init_node("RL_controller",anonymous=True)
        self.use_sim = rospy.get_param("/rl_agent/use_sim")

        if self.use_sim:
            self.rs_image_sub = rospy.Subscriber("/vrep/depth_image",Image,self.depth_CB, queue_size=1)
            self.jointstate_sub = rospy.Subscriber("/vrep/joint_states",JointState,self.jointstate_CB, queue_size=1)
        else:
            self.rs_image_sub = rospy.Subscriber("/camera/depth/image_rect_raw",Image,self.depth_CB, queue_size=1)
            self.jointstate_sub = rospy.Subscriber("/jaco/joint_states",JointState,self.jointstate_CB, queue_size=1)
        self.agent_trigger = rospy.Subscriber("/clock",Clock,self.agent)
        self.pose_pub = rospy.Publisher("/RL_controller/pub_pose",Pose,queue_size=10)

        self.image_buff = []
        self.states_buffersize = 100
        self.joint_state = []
        
    
    def spin(self, rate):
        self.rate = rospy.Rate(rate)
        rospy.spin()


    def depth_CB(self,msg):
        msg_time = msg.header.stamp.to_sec()
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        if self.use_sim:
            data = np.flip(data,0)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1,1,1)
        plt.axis('off')
        #fig.savefig('/home/ljh/Documents/Figure_1.png', bbox_inches='tight',pad_inches=0)
        plt.imshow(data)
        plt.show()
        print("depth image: ",msg_time)
        self.image_buff = [data,msg_time]
    
    def jointstate_CB(self,msg):
        msg_time = msg.header.stamp.to_sec()
        self.joint_state.append([msg.position,msg_time])
        if len(self.joint_state) > self.states_buffersize:
            self.joint_state.pop(0)
            #print("poped")
        print("joint state: ", msg_time, len(self.joint_state)," ",self.joint_state[-1])


    def agent(self,msg):
        self.sensor_fusion()


    def sensor_fusion(self):
        pass



if __name__=="__main__":
    try:
        controller_class = RL_controller()
        controller_class.spin(10)
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")