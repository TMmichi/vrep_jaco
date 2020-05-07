#!/usr/bin/env python3

import os
import time
import datetime
import rospy
import rospkg
import numpy as np
from matplotlib import pyplot as plt
from random import randint, sample
from argparser import ArgParser
from std_msgs.msg import Int8, Float32MultiArray
from sensor_msgs.msg import Image, JointState
from env.env_vrep import JacoVrepEnv
from env.env_real import Real

import yolo


class Data_collector:
    def __init__(self, feedbackRate_=50):
        #Arguments
        parser = ArgParser()
        args = parser.parse_args()
        args.debug = True
        print("DEBUG = ", args.debug)

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
        self.joint_time =  0
        self.num = 0
        self.stop = False
        self.rospack = rospkg.RosPack()
        self.net0 = yolo.load_net(b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/rl_controller/scripts/darknet/cfg/yolov3.cfg", b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/rl_controller/scripts/darknet/cfg/yolov3.weights", 0)
        #self.net1 = yolo.load_net(b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/rl_controller/scripts/darknet/cfg/yolov3.cfg", b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/rl_controller/scripts/darknet/cfg/yolov3.weights", 0)
        #self.net2 = yolo.load_net(b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/rl_controller/scripts/darknet/cfg/yolov3.cfg", b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/rl_controller/scripts/darknet/cfg/yolov3.weights", 0)
        #self.net3 = yolo.load_net(b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/rl_controller/scripts/darknet/cfg/yolov3.cfg", b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/rl_controller/scripts/darknet/cfg/yolov3.weights", 0)
        self.meta = yolo.load_meta(b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/rl_controller/scripts/darknet/cfg/coco.data")
        args.data_path = "/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/data_stategen/"
        os.makedirs(args.data_path,exist_ok=True)

        #ROS settings
        rospy.init_node("Data_collector", anonymous=True)
        self.use_sim = rospy.get_param("/rl_controller/use_sim")
        self.trigger_sub = rospy.Subscriber("key_input", Int8, self.trigger, queue_size=10)
        self.jointstate_sub = rospy.Subscriber("/vrep/joint_states",JointState,self.jointstateCallback, queue_size=1)
        self.pressure_sub = rospy.Subscriber("/vrep/pressure_data",Float32MultiArray,self.pressureCallback, queue_size=1)
        self.depth0_sub = rospy.Subscriber("/vrep/depth_image0",Image,self.depth0Callback, queue_size=1)
        self.image0_sub = rospy.Subscriber("/vrep/rgb_image0",Image,self.image0Callback, queue_size=1)
        self.depth1_sub = rospy.Subscriber("/vrep/depth_image1",Image,self.depth1Callback, queue_size=1)
        self.image1_sub = rospy.Subscriber("/vrep/rgb_image1",Image,self.image1Callback, queue_size=1)
        self.depth2_sub = rospy.Subscriber("/vrep/depth_image2",Image,self.depth2Callback, queue_size=1)
        self.image2_sub = rospy.Subscriber("/vrep/rgb_image2",Image,self.image2Callback, queue_size=1)
        #self.depth3_sub = rospy.Subscriber("/vrep/depth_image3",Image,self.depth3Callback, queue_size=1)
        #self.image3_sub = rospy.Subscriber("/vrep/rgb_image3",Image,self.image3Callback, queue_size=1)

        self.rate = rospy.Rate(feedbackRate_)
        self.period = rospy.Duration(1.0/feedbackRate_)
        args.rate = self.rate
        args.period = self.period
        self.env = JacoVrepEnv(**vars(args)) if self.use_sim else Real(**vars(args))        


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

    def depth0Callback(self,msg):
        self.depthProcess(msg)
    
    def depth1Callback(self,msg):
        self.depthProcess(msg)

    def depth2Callback(self,msg):
        self.depthProcess(msg)

    def depth3Callback(self,msg):
        self.depthProcess(msg)
    
    def depthProcess(self,msg):
        msg_time = round(msg.header.stamp.to_sec(),2)
        width = msg.width
        height = msg.height
        data = np.fromstring(msg.data,dtype=np.uint16)
        data = np.reshape(data,(height,width))
        data = np.flip(data,0)
        self.image_buff = [data,msg_time]
        self.data_buff_temp[0] = self.image_buff

    def image0Callback(self,msg):
        fig_path = b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/vrep_jaco_data/image/figure_rgb1.jpg"
        self.imageProcess(msg,self.net0,fig_path,1)

    def image1Callback(self,msg):
        fig_path = b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/vrep_jaco_data/image/figure_rgb2.jpg"
        self.imageProcess(msg,self.net0,fig_path,2)
    
    def image2Callback(self,msg):
        fig_path = b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/vrep_jaco_data/image/figure_rgb3.jpg"
        self.imageProcess(msg,self.net0,fig_path,3)
    
    def image3Callback(self,msg):
        fig_path = b"/home/ljh/Project/vrep_jaco/vrep_jaco/src/vrep_jaco/vrep_jaco_data/image/figure_rgb4.jpg"
        self.imageProcess(msg,self.net1,fig_path,4)
    
    def imageProcess(self,msg,net,fig_path, index):
        try:
            msg_time = round(msg.header.stamp.to_sec(),2)
            data = np.fromstring(msg.data,dtype=np.uint8)
            data = np.reshape(data,(msg.height,msg.width,3))
            data = np.flip(data,0)
            then = datetime.datetime.now()
            plt.imsave(self.rospack.get_path('vrep_jaco_data')+'/image/figure_rgb'+str(index)+'.jpg',data)
            plt.close()
            r = yolo.detect(net, self.meta, fig_path)
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
            print(data_center)
            print("CALLED TIME: ",datetime.datetime.now() - then)

            self.rgb_image_buff = [data_center, msg_time]
            self.data_buff_temp[3] = self.rgb_image_buff

            if self.joint_start==True and self.pressure_start==True and self.data_time_check(0.075):
                self.data_buff.append(self.data_buff_temp.copy())
        except Exception as e:
            print(e)

    def jointstateCallback(self,msg):
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
                        self.num+=1
        
    def pressureCallback(self,msg):
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
                        self.num+=1
        
    ## 0.05 : 10hz | 0.075 : 20hz 계산
    def data_time_check(self, interval):
        if abs(self.data_buff_temp[0][1]-self.data_buff_temp[1][1])<interval and abs(self.data_buff_temp[1][1]-self.data_buff_temp[2][1])<interval and abs(self.data_buff_temp[0][1]-self.data_buff_temp[2][1])<interval and abs(self.data_buff_temp[3][1]-self.data_buff_temp[0][1])<interval and abs(self.data_buff_temp[3][1]-self.data_buff_temp[1][1])<interval and abs(self.data_buff_temp[3][1]-self.data_buff_temp[2][1])<interval:
            return True
        else:
            return False

    def data_record(self):
        cam_pose0 = [0,-1.75,0.75,-5,0,0]
        cam_pose1 = [1.75,0,0.75,0,-5,90]
        cam_pose2 = [0,0,2.5,103.7,0,180]
        cam_pose3 = [0.5,1.225,2.1,58,-15,170]
        np.save(self.data_path,self.data_buff)
        self.stop = True # if False it records all data
        print("DATA RECORDED")


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
