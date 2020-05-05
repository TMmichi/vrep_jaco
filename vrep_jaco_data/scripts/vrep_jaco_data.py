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
        self.image_buff0 = []
        self.image_buff1 = []
        self.image_buff2 = []
        self.image_buff3 = []
        self.joint_buffersize = 100
        self.joint_state = []
        self.pressure_buffersize = 100
        self.pressure_state = []
        self.rgb_image_buffersize = 5
        self.rgb_image_buff0 = []
        self.rgb_image_buff1 = []
        self.rgb_image_buff2 = []
        self.rgb_image_buff3 = []

        self.data_buff = []
        self.data_buff_temp = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        # [depth_image_data0,depth_image_data1,depth_image_data2,depth_image_data3,
        #  rgb_image_data0,rgb_image_data1,rgb_image_data2,rgb_image_data3,
        #  joint_data,pressure_data]

        self.joint_trigger = True
        self.pressure_trigger = True

        self.joint_start = False # at first
        self.pressure_start = False # at first
        self.depth0 = False # at first
        self.depth1 = False # at first
        self.depth2 = False # at first
        self.depth3 = False # at first
        self.rgb0 = False # at first
        self.rgb1 = False # at first
        self.rgb2 = False # at first
        self.rgb3 = False # at first

        self.rospack = rospkg.RosPack()
        self.joint_time =  0
        self.num = 0

        self.stop = False

        ## absolute path needed
        self.net = yolo.load_net(b"/home/kth/Desktop/ws/lab/vrep_jaco_ws/src/vrep_jaco_data/scripts/darknet/cfg/yolov3-tiny.cfg", b"/home/kth/Desktop/ws/lab/vrep_jaco_ws/src/vrep_jaco_data/scripts/darknet/yolov3-tiny.weights", 0)
        self.meta = yolo.load_meta(b"/home/kth/Desktop/ws/lab/vrep_jaco_ws/src/vrep_jaco_data/scripts/darknet/cfg/coco.data")

        self.jointstate_sub = rospy.Subscriber("/vrep/joint_states",JointState,self.jointstate_CB, queue_size=1)
        self.pressure_sub = rospy.Subscriber("/vrep/pressure_data",Float32MultiArray,self.pressure_CB, queue_size=1)
        self.rs_image_sub0 = rospy.Subscriber("/vrep/depth_image0",Image,self.depth_CB0, queue_size=1)
        self.rgb_rs_image_sub0 = rospy.Subscriber("/vrep/rgb_image0",Image,self.rgb_CB0, queue_size=1)
        self.rs_image_sub1 = rospy.Subscriber("/vrep/depth_image1",Image,self.depth_CB1, queue_size=1)
        self.rgb_rs_image_sub1 = rospy.Subscriber("/vrep/rgb_image1",Image,self.rgb_CB1, queue_size=1)
        self.rs_image_sub2 = rospy.Subscriber("/vrep/depth_image2",Image,self.depth_CB2, queue_size=1)
        self.rgb_rs_image_sub2 = rospy.Subscriber("/vrep/rgb_image2",Image,self.rgb_CB2, queue_size=1)
        self.rs_image_sub3 = rospy.Subscriber("/vrep/depth_image3",Image,self.depth_CB3, queue_size=1)
        self.rgb_rs_image_sub3 = rospy.Subscriber("/vrep/rgb_image3",Image,self.rgb_CB3, queue_size=1)

        # self.center = Point()
        # self.center_pub = rospy.Publisher("/yolo",Point)
    
    def depth_CB0(self,msg):
        # self.joint_trigger = True
        # self.pressure_trigger = True
        self.depth0 = True
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        self.image_buff0 = [data,msg_time]
        self.data_buff_temp[0] = self.image_buff0
        # print("depth0 : ",self.depth0)

    def depth_CB1(self,msg):
        # self.joint_trigger = True
        # self.pressure_trigger = True
        self.depth1 = True
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        self.image_buff1 = [data,msg_time]
        self.data_buff_temp[1] = self.image_buff1
        # print("depth : ",msg_time)
        # print("depth1 : ",self.depth1)


    def depth_CB2(self,msg):
        # self.joint_trigger = True
        # self.pressure_trigger = True
        self.depth2 = True
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        self.image_buff2 = [data,msg_time]
        self.data_buff_temp[2] = self.image_buff2
        # print("depth : ",msg_time)
        # print("depth2 : ",self.depth2)


    def depth_CB3(self,msg):
        # self.joint_trigger = True
        # self.pressure_trigger = True
        self.depth3 = True
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        self.image_buff3 = [data,msg_time]
        self.data_buff_temp[3] = self.image_buff3
        # print("depth : ",msg_time)
        # print("depth3 : ",self.depth3)



    def rgb_CB0(self,msg):
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint8)
        data = np.reshape(data,(height,width,3))
        data = np.flip(data,0)
        plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure_rgb0.jpg',data)
        plt.close()
        print("gogo")
        r = yolo.detect(self.net, self.meta, b"/home/kth/Desktop/ws/lab/vrep_jaco_ws/src/vrep_jaco_data/image/figure_rgb0.jpg")
        if(len(r)==0):
            data_center = [[-1,-1]]
        else:
            print(r[0][0])
            data_center = []
            a = 0
            for i in range(len(r)):
                if a==0 and r[i][0]==b'cup':
                    a+=1
                    data_center = [[r[i][2][0],r[i][2][1]]]
                elif a!=0 and r[i][0]==b'cup':
                    data_center = data_center+[[r[i][2][0],r[i][2][1]]]
            if len(data_center)==0:
                data_center = [[-1,-1]]
            
        self.rgb_image_buff0 = [data_center, msg_time]
        self.data_buff_temp[4] = self.rgb_image_buff0
        self.rgb0 = True
        ## should fix time
        # if self.joint_start==True and self.pressure_start==True and self.data_time_check(0.075):
        #     self.data_buff.append(self.data_buff_temp.copy())
 
    def rgb_CB1(self,msg):
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint8)
        data = np.reshape(data,(height,width,3))
        data = np.flip(data,0)
        plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure_rgb1.jpg',data)
        plt.close()
        r = yolo.detect(self.net, self.meta, b"/home/kth/Desktop/ws/lab/vrep_jaco_ws/src/vrep_jaco_data/image/figure_rgb1.jpg")
        if(len(r)==0):
            data_center = [[-1,-1]]
        else:
            data_center = []
            a = 0
            for i in range(len(r)):
                if a==0 and r[i][0]==b'cup':
                    a+=1
                    data_center = [[r[i][2][0],r[i][2][1]]]
                elif a!=0 and r[i][0]==b'cup':
                    data_center = data_center+[[r[i][2][0],r[i][2][1]]]
            if len(data_center)==0:
                data_center = [[-1,-1]]
            
        self.rgb_image_buff1 = [data_center, msg_time]
        self.data_buff_temp[5] = self.rgb_image_buff1
        self.rgb1 = True
        ## should fix time
        # if self.joint_start==True and self.pressure_start==True and self.data_time_check(0.075):
        #     self.data_buff.append(self.data_buff_temp.copy())
    
    def rgb_CB2(self,msg):
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint8)
        data = np.reshape(data,(height,width,3))
        data = np.flip(data,0)
        plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure_rgb2.jpg',data)
        plt.close()
        r = yolo.detect(self.net, self.meta, b"/home/kth/Desktop/ws/lab/vrep_jaco_ws/src/vrep_jaco_data/image/figure_rgb2.jpg")
        if(len(r)==0):
            data_center = [[-1,-1]]
        else:
            data_center = []
            a = 0
            for i in range(len(r)):
                if a==0 and r[i][0]==b'cup':
                    a+=1
                    data_center = [[r[i][2][0],r[i][2][1]]]
                elif a!=0 and r[i][0]==b'cup':
                    data_center = data_center+[[r[i][2][0],r[i][2][1]]]
            if len(data_center)==0:
                data_center = [[-1,-1]]
            
        self.rgb_image_buff2 = [data_center, msg_time]
        self.data_buff_temp[6] = self.rgb_image_buff2
        self.rgb2 = True
        ## should fix time
        # if self.joint_start==True and self.pressure_start==True and self.data_time_check(0.075):
        #     self.data_buff.append(self.data_buff_temp.copy())
    
    def rgb_CB3(self,msg):
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint8)
        data = np.reshape(data,(height,width,3))
        data = np.flip(data,0)
        plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure_rgb3.jpg',data)
        plt.close()
        r = yolo.detect(self.net, self.meta, b"/home/kth/Desktop/ws/lab/vrep_jaco_ws/src/vrep_jaco_data/image/figure_rgb3.jpg")
        if(len(r)==0):
            data_center = [[-1,-1]]
        else:
            data_center = []
            a = 0
            for i in range(len(r)):
                if a==0 and r[i][0]==b'cup':
                    a+=1
                    data_center = [[r[i][2][0],r[i][2][1]]]
                elif a!=0 and r[i][0]==b'cup':
                    data_center = data_center+[[r[i][2][0],r[i][2][1]]]
            if len(data_center)==0:
                data_center = [[-1,-1]]
            
        self.rgb_image_buff3 = [data_center, msg_time]
        self.data_buff_temp[7] = self.rgb_image_buff3
        self.rgb3 = True
        ## should fix time
        # if self.joint_start==True and self.pressure_start==True and self.data_time_check(0.075):
        #     self.data_buff.append(self.data_buff_temp.copy())
    
    def spin(self, rate):
        self.rate = rospy.Rate(rate)
        rospy.spin()

    # def depth_CB(self,msg):
    #     # self.joint_trigger = True
    #     # self.pressure_trigger = True

    #     msg_time = round(msg.header.stamp.to_sec(),2)
    #     width = msg.width
    #     height = msg.height
    #     data = np.fromstring(msg.data,dtype=np.uint16)
    #     data = np.reshape(data,(height,width))
    #     data = np.flip(data,0)
    #     self.image_buff = [data,msg_time]
    #     self.data_buff_temp[0] = self.image_buff
    #     # print("depth : ",msg_time)

    # def rgb_CB(self,msg):
    #     msg_time = round(msg.header.stamp.to_sec(),2)
    #     width = msg.width
    #     height = msg.height
    #     data = np.fromstring(msg.data,dtype=np.uint8)
    #     data = np.reshape(data,(height,width,3))
    #     data = np.flip(data,0)
    #     plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure_rgb.jpg',data)
    #     plt.close()
    #     r = yolo.detect(self.net, self.meta, b"/home/kimtaehan/Desktop/vrep_jaco_2/src/vrep_jaco_data/image/figure_rgb.jpg")
    #     if(len(r)==0):
    #         data_center = [[-1,-1]]
    #     else:
    #         data_center = []
    #         for i in range(len(r)):
    #             a = 0
    #             if a==0 and r[i][0]==b'cup':
    #                 a+=1
    #                 data_center = [[r[i][2][0],r[i][2][1]]]
    #             elif a!=0 and r[i][0]==b'cup':
    #                 data_center = data_center+[[r[i][2][0],r[i][2][1]]]
    #         if len(data_center)==0:
    #             data_center = [[-1,-1]]
            
    #     self.rgb_image_buff = [data_center, msg_time]
    #     self.data_buff_temp[3] = self.rgb_image_buff

    #     if self.joint_start==True and self.pressure_start==True and self.data_time_check(0.075):
    #         self.data_buff.append(self.data_buff_temp.copy())
    #     # print("rgb : ",msg_time)
    #     # print("rgb : ",data_center)
    #     # print('rgb : ',data_center)
        

    #     # self.rgb_image_buff = [data, msg_time]
    #     # self.data_buff_temp[3] = self.rgb_image_buff
 
    #     # print(len(r))
    #     # self.center.x = r[0][2][0]
    #     # self.center.y = r[0][2][1]
    #     # self.center.z = 0
    #     # self.center_pub.publish(self.center)

    def jointstate_CB(self,msg):
        if self.stop == False:
            if self.joint_trigger:
                msg_time = round(msg.header.stamp.to_sec(),2)
                # print("joint : ",msg_time)
                self.joint_state.append([msg.position,msg_time])
                if len(self.joint_state) > self.joint_buffersize:
                    self.joint_state.pop(0)
                self.data_buff_temp[8] = self.joint_state[-1]
                self.joint_start = True
                self.joint_trigger = False
                if not self.pressure_trigger:
                    self.pressure_trigger = True
                    if self.pressure_start == True and self.depth0 == True and self.depth1 == True and self.depth2 == True and self.depth3 == True and self.rgb0 == True and self.rgb1 == True and self.rgb2 == True and self.rgb3 == True and self.data_time_check(0.06):
                        self.data_buff.append(self.data_buff_temp.copy())
                        # plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure'+str(self.num)+'.png',self.image_buff[0],cmap='gray')
                        # plt.close()
                        # self.num+=1
                    if msg_time > 5: # saving time control
                        self.data_record()
        

    def pressure_CB(self,msg):
        if self.stop == False:
            if self.pressure_trigger:
                msg_time = round(msg.data[0],2)
                # print("pressure : ",msg_time)
                self.pressure_state.append([msg.data[1:],msg_time])
                if len(self.pressure_state) > self.pressure_buffersize:
                    self.pressure_state.pop(0)
                self.data_buff_temp[9] = self.pressure_state[-1]
                self.pressure_start = True
                self.pressure_trigger = False
                if not self.joint_trigger:
                    self.joint_trigger = True
                    if self.joint_start == True and self.depth0 == True and self.depth1 == True and self.depth2 == True and self.depth3 == True and self.rgb0 == True and self.rgb1 == True and self.rgb2 == True and self.rgb3 == True and self.data_time_check(0.06):
                        self.data_buff.append(self.data_buff_temp.copy())
                        print("hi")
                        # plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure'+str(self.num)+'.png',self.image_buff[0],cmap='gray')
                        # plt.close()
                        # self.num+=1
                    if msg_time > 5: # saving time control
                        self.data_record()
        
    ## 0.05 : 10hz | 0.075 : 20hz 계산
    def data_time_check(self, interval):
        maxima = self.data_buff_temp[0][1]
        minima = self.data_buff_temp[0][1]
        for i in range(10):
            if maxima<self.data_buff_temp[i][1]:
                maxima=self.data_buff_temp[i][1]
            if minima>self.data_buff_temp[i][1]:
                minima=self.data_buff_temp[i][1]
        if abs(maxima-minima)<interval:
            return True
        else:
            return False

    def data_record(self):
        np.save(self.rospack.get_path('vrep_jaco_data')+"/data/dummy_data",self.data_buff)
        # self.stop = True # if False it records all data
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
        vjd.spin(20)
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")





    # net = yolo.load_net(b"/home/kimtaehan/Desktop/yolo_darknet/darknet/cfg/yolov3.cfg", b"/home/kimtaehan/Desktop/yolo_darknet/darknet/yolov3.weights", 0)
    # meta = yolo.load_meta(b"/home/kimtaehan/Desktop/yolo_darknet/darknet/cfg/coco.data")
    # r = yolo.detect(net, meta, b"/home/kimtaehan/Desktop/yolo_darknet/darknet/data/dog.jpg")
    # print (r[0][2][0], r[0][2][1])