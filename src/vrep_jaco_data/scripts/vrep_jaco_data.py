#!/usr/bin/env python3
import time
#import math
#import cv2 as cv
#from cv_bridge import CvBridge, CvBridgeError
#from matplotlib import pyplot as plt
import numpy as np
#import vae_util

import rospy
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState


class Vrep_jaco_data:
    def __init__(self):
        rospy.init_node("vrep_jaco_data",anonymous=True)
        self.rs_image_sub = rospy.Subscriber("/vrep/depth_image",Image,self.depth_CB, queue_size=1)
        self.jointstate_sub = rospy.Subscriber("/jaco/joint_states",JointState,self.jointstate_CB, queue_size=1)
        self.pressure_sub = rospy.Subscriber("/vrep/pressure_data",Float32MultiArray,self.pressure_CB, queue_size=1)
        
        self.image_buffersize = 5
        self.image_buff = []
        self.joint_buffersize = 100
        self.joint_state = []
        self.pressure_buffersize = 100
        self.pressure_state = []

        self.data_buff = []
        self.data_buff_temp = [0,0,0]

        self.depth_trigger = False
        self.joint_trigger = False
        self.pressure_trigger = False
        
    
    def spin(self, rate):
        self.rate = rospy.Rate(rate)
        rospy.spin()

    def depth_CB(self,msg):
        self.depth_trigger = True
        self.joint_trigger = True
        self.pressure_trigger = True

        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        '''
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1,1,1)
        plt.axis('off')
        #fig.savefig('/home/ljh/Documents/Figure_1.png', bbox_inches='tight',pad_inches=0)
        plt.imshow(data)
        plt.show()'''
        print("depth image: ",msg_time)
        self.image_buff = [data,msg_time]
        self.data_buff_temp[0] = self.image_buff

    
    def jointstate_CB(self,msg):
        if self.joint_trigger:
            msg_time = round(msg.header.stamp.to_sec(),2)
            self.joint_state.append([msg.position,msg_time])
            if len(self.joint_state) > self.joint_buffersize:
                self.joint_state.pop(0)
            print("joint state: ", msg_time)
            self.data_buff_temp[1] = self.joint_state[-1]
            self.joint_trigger = False
            if not self.pressure_trigger:
                self.data_buff.append(self.data_buff_temp)
                if msg_time > 3:
                    self.data_record()
        

    def pressure_CB(self,msg):
        if self.pressure_trigger:
            msg_time = round(msg.data[0],2)
            self.pressure_state.append([msg.data[1:],msg_time])
            if len(self.pressure_state) > self.pressure_buffersize:
                self.pressure_state.pop(0)
            print("pressure state: ", msg_time)
            self.data_buff_temp[2] = self.pressure_state[-1]
            self.pressure_trigger = False
            if not self.joint_trigger:
                self.data_buff.append(self.data_buff_temp)
                if msg_time > 3:
                    self.data_record()

    def data_record(self):
        np.save("/home/ljh/Project/vrep_jaco/src/vrep_jaco_data/data/dummy_data",self.data_buff)
        print("DATA RECORDED")
        raise rospy.ROSInterruptException


if __name__=="__main__":
    try:
        vjd = Vrep_jaco_data()
        vjd.spin(10)
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")
