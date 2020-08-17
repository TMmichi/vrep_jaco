#!/usr/bin/env python3
import time
from matplotlib import pyplot as plt
import numpy as np
import rospy
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
import rospkg

from datetime import datetime

class Vrep_jaco_data:
    def __init__(self):
        rospy.init_node("vrep_jaco_data",anonymous=True)
        
        self.image_buffersize = 5
        self.image_buff = []
        self.joint_buffersize = 100
        self.joint_state = []
        self.pressure_buffersize = 100
        self.pressure_state = []
        self.rgb_image_buffersize = 5
        self.rgb_image_buff = []

        self.data_buff = []
        self.data_buff_temp = [[0,0],[0,0],[0,0],[0,0]]

        self.joint_trigger = True
        self.pressure_trigger = True

        self.joint_start = False # at first
        self.pressure_start = False # at first

        self.rospack = rospkg.RosPack()
        self.joint_time =  0
        self.num = 0

        self.stop = False
        
        self.rs_image_sub = rospy.Subscriber("/vrep/depth_image0",Image,self.depth_CB0, queue_size=10)
        self.rs_image_sub = rospy.Subscriber("/vrep/depth_image1",Image,self.depth_CB1, queue_size=10)
        self.rs_image_sub = rospy.Subscriber("/vrep/depth_image2",Image,self.depth_CB2, queue_size=10)
        self.rs_image_sub = rospy.Subscriber("/vrep/depth_image3",Image,self.depth_CB3, queue_size=10)
        
        # self.jointstate_sub = rospy.Subscriber("/vrep/joint_states",JointState,self.jointstate_CB, queue_size=1)
        # self.pressure_sub = rospy.Subscriber("/vrep/pressure_data",Float32MultiArray,self.pressure_CB, queue_size=1)
    
    def spin(self, rate):
        self.rate = rospy.Rate(rate)
        rospy.spin()

    def depth_CB0(self,msg):
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        self.image_buff = [data.tolist(),msg_time]
        self.data_buff_temp[0] = self.image_buff

    def depth_CB1(self,msg):
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        self.image_buff = [data.tolist(),msg_time]
        self.data_buff_temp[1] = self.image_buff

    def depth_CB2(self,msg):
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        self.image_buff = [data.tolist(),msg_time]
        self.data_buff_temp[2] = self.image_buff

    def depth_CB3(self,msg):
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        self.image_buff = [data.tolist(),msg_time]
        self.data_buff_temp[3] = self.image_buff
        if self.data_time_check(0.051):
            self.data_record()

    # def jointstate_CB(self,msg):
    #     if self.stop == False:
    #         if self.joint_trigger:
    #             msg_time = round(msg.header.stamp.to_sec(),2)
    #             # print("joint : ",msg_time)
    #             self.joint_state.append([msg.position,msg_time])
    #             if len(self.joint_state) > self.joint_buffersize:
    #                 self.joint_state.pop(0)
    #             self.data_buff_temp[1] = self.joint_state[-1]
    #             self.joint_start = True
    #             self.joint_trigger = False
    #             if not self.pressure_trigger:
    #                 self.pressure_trigger = True
    #                 if self.pressure_start == True and self.data_time_check(0.075):
    #                     self.data_buff.append(self.data_buff_temp.copy())
    #                     plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure'+str(self.num)+'.png',self.image_buff[0],cmap='gray')
    #                     plt.close()
    #                     self.num+=1
    #                 if msg_time > 10: # saving time control
    #                     self.data_record()
        
    # def pressure_CB(self,msg):
    #     if self.stop == False:
    #         if self.pressure_trigger:
    #             msg_time = round(msg.data[0],2)
    #             # print("pressure : ",msg_time)
    #             self.pressure_state.append([msg.data[1:],msg_time])
    #             if len(self.pressure_state) > self.pressure_buffersize:
    #                 self.pressure_state.pop(0)
    #             self.data_buff_temp[2] = self.pressure_state[-1]
    #             self.pressure_start = True
    #             self.pressure_trigger = False
    #             if not self.joint_trigger:
    #                 self.joint_trigger = True
    #                 if self.joint_start == True and self.data_time_check(0.075):
    #                     self.data_buff.append(self.data_buff_temp.copy())
    #                     plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure'+str(self.num)+'.png',self.image_buff[0],cmap='gray')
    #                     plt.close()
    #                     self.num+=1
    #                 if msg_time > 10: # saving time control
    #                     self.data_record()
        
    ## 0.05 : 10hz | 0.075 : 20hz 계산
    def data_time_check(self, interval):
        if abs(self.data_buff_temp[0][1]-self.data_buff_temp[1][1])<interval and abs(self.data_buff_temp[1][1]-self.data_buff_temp[2][1])<interval and abs(self.data_buff_temp[0][1]-self.data_buff_temp[2][1])<interval and abs(self.data_buff_temp[3][1]-self.data_buff_temp[0][1])<interval and abs(self.data_buff_temp[3][1]-self.data_buff_temp[1][1])<interval and abs(self.data_buff_temp[3][1]-self.data_buff_temp[2][1])<interval:
            return True
        else:
            return False

    def data_record(self):
        self.data_buff=self.data_buff_temp
        # now = datetime.now()
        # np.save(self.rospack.get_path('vrep_jaco_data')+"/data/"+str(now.year)+str(now.month)+str(now.day)+str(now.hour)+str(now.minute),self.data_buff)
        np.save(self.rospack.get_path('vrep_jaco_data')+"/data/data_before/"+str(self.num),self.data_buff)
        self.num+=1
        print("DATA RECORDED")


if __name__=="__main__":
    try:
        vjd = Vrep_jaco_data()
        vjd.spin(10)
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")
