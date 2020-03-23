#!/usr/bin/env python3
import time
#import math
#import cv2 as cv
#from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt
import numpy as np
#import vae_util

import rospy
import rospkg
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
# from PIL import Image

import yolo


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
        self.net = yolo.load_net(b"/home/kimtaehan/Desktop/vrep_jaco_2/src/vrep_jaco_data/scripts/darknet/cfg/yolov3.cfg", b"/home/kimtaehan/Desktop/vrep_jaco_2/src/vrep_jaco_data/scripts/darknet/yolov3.weights", 0)
        self.meta = yolo.load_meta(b"/home/kimtaehan/Desktop/vrep_jaco_2/src/vrep_jaco_data/scripts/darknet/cfg/coco.data")

        self.rs_image_sub = rospy.Subscriber("/vrep/depth_image",Image,self.depth_CB, queue_size=1)
        self.jointstate_sub = rospy.Subscriber("/vrep/joint_states",JointState,self.jointstate_CB, queue_size=1)
        self.pressure_sub = rospy.Subscriber("/vrep/pressure_data",Float32MultiArray,self.pressure_CB, queue_size=1)
        self.rgb_rs_image_sub = rospy.Subscriber("/vrep/rgb_image",Image,self.rgb_CB, queue_size=1)

        # self.center = Point()
        # self.center_pub = rospy.Publisher("/yolo",Point)
    
        
    
    def spin(self, rate):
        self.rate = rospy.Rate(rate)
        rospy.spin()

    def depth_CB(self,msg):
        # self.joint_trigger = True
        # self.pressure_trigger = True

        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        self.image_buff = [data,msg_time]
        self.data_buff_temp[0] = self.image_buff
        # print("depth : ",msg_time)

    def rgb_CB(self,msg):
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint8)
        data = np.reshape(data,(height,width,3))
        data = np.flip(data,0)
        plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure_rgb.jpg',data)
        plt.close()
        r = yolo.detect(self.net, self.meta, b"/home/kimtaehan/Desktop/vrep_jaco_2/src/vrep_jaco_data/image/figure_rgb.jpg")
        if(len(r)==0):
            data_center = [[-1,-1]]
        else:
            data_center = []
            for i in range(len(r)):
                a = 0
                if a==0 and r[i][0]==b'cup':
                    a+=1
                    data_center = [[r[i][2][0],r[i][2][1]]]
                elif a!=0 and r[i][0]==b'cup':
                    data_center = data_center+[[r[i][2][0],r[i][2][1]]]
            if len(data_center)==0:
                data_center = [[-1,-1]]
            
        self.rgb_image_buff = [data_center, msg_time]
        self.data_buff_temp[3] = self.rgb_image_buff

        if self.joint_start==True and self.pressure_start==True and self.data_time_check(0.075):
            self.data_buff.append(self.data_buff_temp.copy())
        # print("rgb : ",msg_time)
        # print("rgb : ",data_center)
        # print('rgb : ',data_center)
        

        # self.rgb_image_buff = [data, msg_time]
        # self.data_buff_temp[3] = self.rgb_image_buff
 
        # print(len(r))
        # self.center.x = r[0][2][0]
        # self.center.y = r[0][2][1]
        # self.center.z = 0
        # self.center_pub.publish(self.center)

    def jointstate_CB(self,msg):
        if self.stop == False:
            if self.joint_trigger:
                msg_time = round(msg.header.stamp.to_sec(),2)
                # print("joint : ",msg_time)
                self.joint_state.append([msg.position,msg_time])
                if len(self.joint_state) > self.joint_buffersize:
                    self.joint_state.pop(0)
                self.data_buff_temp[1] = self.joint_state[-1]
                self.joint_start = True
                self.joint_trigger = False
                if not self.pressure_trigger:
                    self.pressure_trigger = True
                    if self.pressure_start == True and self.data_time_check(0.075):
                        self.data_buff.append(self.data_buff_temp.copy())
                        plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure'+str(self.num)+'.png',self.image_buff[0],cmap='gray')
                        plt.close()
                        self.num+=1
                    if msg_time > 10: # saving time control
                        self.data_record()
        

    def pressure_CB(self,msg):
        if self.stop == False:
            if self.pressure_trigger:
                msg_time = round(msg.data[0],2)
                # print("pressure : ",msg_time)
                self.pressure_state.append([msg.data[1:],msg_time])
                if len(self.pressure_state) > self.pressure_buffersize:
                    self.pressure_state.pop(0)
                self.data_buff_temp[2] = self.pressure_state[-1]
                self.pressure_start = True
                self.pressure_trigger = False
                if not self.joint_trigger:
                    self.joint_trigger = True
                    if self.joint_start == True and self.data_time_check(0.075):
                        self.data_buff.append(self.data_buff_temp.copy())
                        plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure'+str(self.num)+'.png',self.image_buff[0],cmap='gray')
                        plt.close()
                        self.num+=1
                    if msg_time > 10: # saving time control
                        self.data_record()
        
    ## 0.05 : 10hz | 0.075 : 20hz 계산
    def data_time_check(self, interval):
        if abs(self.data_buff_temp[0][1]-self.data_buff_temp[1][1])<interval and abs(self.data_buff_temp[1][1]-self.data_buff_temp[2][1])<interval and abs(self.data_buff_temp[0][1]-self.data_buff_temp[2][1])<interval and abs(self.data_buff_temp[3][1]-self.data_buff_temp[0][1])<interval and abs(self.data_buff_temp[3][1]-self.data_buff_temp[1][1])<interval and abs(self.data_buff_temp[3][1]-self.data_buff_temp[2][1])<interval:
            return True
        else:
            return False

    def data_record(self):
        np.save(self.rospack.get_path('vrep_jaco_data')+"/data/dummy_data",self.data_buff)
        self.stop = True # if False it records all data
        print("DATA RECORDED")
        # # raise rospy.ROSInterruptException

    ###################################################################
    ################################################################### 
    # def testing(self):
    #     r = yolo.detect(self.net, self.meta, b"/home/kimtaehan/Desktop/yolo_darknet/darknet/data/dog.jpg")
    #     print(r[0][2][0], r[0][2][1])



if __name__=="__main__":
    try:
        vjd = Vrep_jaco_data()
        vjd.spin(10)
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")





    # net = yolo.load_net(b"/home/kimtaehan/Desktop/yolo_darknet/darknet/cfg/yolov3.cfg", b"/home/kimtaehan/Desktop/yolo_darknet/darknet/yolov3.weights", 0)
    # meta = yolo.load_meta(b"/home/kimtaehan/Desktop/yolo_darknet/darknet/cfg/coco.data")
    # r = yolo.detect(net, meta, b"/home/kimtaehan/Desktop/yolo_darknet/darknet/data/dog.jpg")
    # print (r[0][2][0], r[0][2][1])