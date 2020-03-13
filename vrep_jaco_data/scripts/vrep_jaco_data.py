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
# from PIL import Image

import yolo


class Vrep_jaco_data:
    def __init__(self):
        rospy.init_node("vrep_jaco_data",anonymous=True)
        self.rs_image_sub = rospy.Subscriber("/vrep/depth_image",Image,self.depth_CB, queue_size=1)
        self.jointstate_sub = rospy.Subscriber("/vrep/joint_states",JointState,self.jointstate_CB, queue_size=1)
        self.pressure_sub = rospy.Subscriber("/vrep/pressure_data",Float32MultiArray,self.pressure_CB, queue_size=1)
        self.rgb_rs_image_sub = rospy.Subscriber("/vrep/rgb_image",Image,self.rgb_CB, queue_size=1)

        self.image_buffersize = 5
        self.image_buff = []
        self.joint_buffersize = 100
        self.joint_state = []
        self.pressure_buffersize = 100
        self.pressure_state = []
        self.rgb_image_buffersize = 5
        self.rgb_image_buff = []

        self.data_buff = []
        self.data_buff_temp = [[0,0],0,0,0]

        # self.depth_trigger = False
        # self.joint_trigger = False
        # self.pressure_trigger = False

        self.depth_trigger = True
        self.joint_trigger = True
        self.pressure_trigger = True

        self.joint_start = False # at first
        self.pressure_start = False # at first

        self.rospack = rospkg.RosPack()
        self.joint_time =  0
        self.num = 0

        self.stop = False

        # self.net = yolo.load_net("/home/kimtaehan/Desktop/vrep_jaco_2/src/vrep_jaco_data/scripts/darknet/cfg/yolov3.cfg", "/home/kimtaehan/Desktop/vrep_jaco_2/src/vrep_jaco_data/scripts/darknet/yolov3.weights", 0)
        # self.meta = yolo.load_meta("/home/kimtaehan/Desktop/vrep_jaco_2/src/vrep_jaco_data/scripts/darknet/cfg/coco.data")
    
        
    
    def spin(self, rate):
        self.rate = rospy.Rate(rate)
        rospy.spin()

    def depth_CB(self,msg):
        # self.depth_trigger = True
        # self.joint_trigger = True
        # self.pressure_trigger = True

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

        # print("depth image: ",msg_time)
        self.image_buff = [data,msg_time]
        self.data_buff_temp[0] = self.image_buff

        #########################################################
        #########################################################

        # if (msg_time-self.joint_time)<0.05:
        #     plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure'+str(self.num),data,cmap='gray')
        #     plt.close()

        #########################################################
        #########################################################
        '''# image to npy converter
        filename = 'image-test'
        img = Image.open( filename + '.png' ) # PIL
        data = np.array( img, dtype='uint8' )
        np.save( filename + '.npy', data)

        # visually testing our output
        img_array = np.load(filename + '.npy')
        plt.imshow(img_array) '''
        #########################################################
        #########################################################


    
    # def jointstate_CB(self,msg):
    #     if self.joint_trigger:
    #         msg_time = round(msg.header.stamp.to_sec(),2)
    #         self.joint_state.append([msg.position,msg_time])
    #         if len(self.joint_state) > self.joint_buffersize:
    #             self.joint_state.pop(0)
    #         print("joint state: ", msg_time)
    #         self.data_buff_temp[1] = self.joint_state[-1]
    #         self.joint_trigger = False
    #         if not self.pressure_trigger:
    #             self.data_buff.append(self.data_buff_temp)
    #             if msg_time > 3:
    #                 self.data_record()

    # def pressure_CB(self,msg):
    #     if self.pressure_trigger:
    #         msg_time = round(msg.data[0],2)
    #         self.pressure_state.append([msg.data[1:],msg_time])
    #         if len(self.pressure_state) > self.pressure_buffersize:
    #             self.pressure_state.pop(0)
    #         print("pressure state: ", msg_time)
    #         self.data_buff_temp[2] = self.pressure_state[-1]
    #         self.pressure_trigger = False
    #         if not self.joint_trigger:
    #             self.data_buff.append(self.data_buff_temp)
    #             if msg_time > 3:
    #                 self.data_record()

    # def data_record(self):
    #     np.save("/home/ljh/Project/vrep_jaco/src/vrep_jaco_data/data/dummy_data",self.data_buff)
    #     print("DATA RECORDED")
    #     raise rospy.ROSInterruptException

    ###################################################################
    ###################################################################    
    def rgb_CB(self,msg):
        # self.depth_trigger = True
        # self.joint_trigger = True
        # self.pressure_trigger = True

        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        print(data)
        # data = np.reshape(data,(height,width))
        # data = np.flip(data,0)
        # plt.imshow(data)
        # plt.show()
        '''
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1,1,1)
        plt.axis('off')
        #fig.savefig('/home/ljh/Documents/Figure_1.png', bbox_inches='tight',pad_inches=0)
        plt.imshow(data)
        plt.show()'''
        # output = []
        # # print("depth image: ",msg_time)
        # r = yolo.detect(self.net, self.meta, data)
        # for i in len(r):
        #     if r[i][0]=='cup':
        #         output.append([r[i][2][0],r[i][2][1]])
        # self.rgb_image_buff = [output,msg_time]
        # self.data_buff_temp[3] = self.rgb_image_buff





    def jointstate_CB(self,msg):
        if self.stop == False:
            if self.joint_trigger:
                msg_time = round(msg.header.stamp.to_sec(),2)
                self.joint_state.append([msg.position,msg_time])
                if len(self.joint_state) > self.joint_buffersize:
                    self.joint_state.pop(0)
                # print("joint state: ", msg_time)
                self.data_buff_temp[1] = self.joint_state[-1]
                # print("joint state: ", self.data_buff_temp[1][1])
                self.joint_start = True
                self.joint_trigger = False
                if not self.pressure_trigger:
                    self.pressure_trigger = True
                    # print("joint_check_time : ",self.data_time_check())
                    # print(self.pressure_start == True and self.data_time_check())
                    if self.pressure_start == True and self.data_time_check():
                        # self.data_buff.append(self.data_buff_temp.copy())
                        # print("joint data_buff : ",self.data_buff[-1][1][1])
                        plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure'+str(self.num)+'.png',self.image_buff[0],cmap='gray')
                        self.num+=1
                        plt.close()
                    if msg_time > 0: #10:
                        self.data_record()

    def pressure_CB(self,msg):
        if self.stop == False:
            if self.pressure_trigger:
                msg_time = round(msg.data[0],2)
                self.pressure_state.append([msg.data[1:],msg_time])
                if len(self.pressure_state) > self.pressure_buffersize:
                    self.pressure_state.pop(0)
                # print("pressure state: ", msg_time)
                self.data_buff_temp[2] = self.pressure_state[-1]
                # print("pressure state: ", self.data_buff_temp[2][1])
                self.pressure_start = True
                self.pressure_trigger = False
                if not self.joint_trigger:
                    self.joint_trigger = True
                    # print("pressure_check_time : ",self.data_time_check())
                    # print(self.joint_start == True and self.data_time_check())
                    if self.joint_start == True and self.data_time_check():
                        # self.data_buff.append(self.data_buff_temp.copy())
                        # print("pressure data_buff : ",self.data_buff[0][2][1])
                        plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure'+str(self.num)+'.png',self.image_buff[0],cmap='gray')
                        self.num+=1
                        plt.close()
                    if msg_time > 0: #10:
                        self.data_record()

    def data_time_check(self):
        if abs(self.data_buff_temp[0][1]-self.data_buff_temp[1][1])<0.05 and abs(self.data_buff_temp[1][1]-self.data_buff_temp[2][1])<0.05 and abs(self.data_buff_temp[0][1]-self.data_buff_temp[2][1])<0.05:
            return True
        else:
            return False

    def data_record(self):
        # print(self.data_buff[0][1][1])
        # print(self.data_buff[-1][1][1])
        np.save(self.rospack.get_path('vrep_jaco_data')+"/data/dummy_data",self.data_buff)
        self.stop = True
        print("DATA RECORDED")
        # raise rospy.ROSInterruptException

    ###################################################################
    ################################################################### 




if __name__=="__main__":
    try:
        vjd = Vrep_jaco_data()
        vjd.spin(10)
    except rospy.ROSInternalException:
        rospy.loginfo("node terminated.")
    # net = yolo.load_net("/home/kimtaehan/Desktop/yolo_darknet/darknet/cfg/yolov3.cfg", "/home/kimtaehan/Desktop/yolo_darknet/darknet/yolov3.weights", 0)
    # meta = yolo.load_meta("/home/kimtaehan/Desktop/yolo_darknet/darknet/cfg/coco.data")
    # r = yolo.detect(net, meta, "/home/kimtaehan/Desktop/yolo_darknet/darknet/data/dog.jpg")
    # print r[0][2][0], r[0][2][1]