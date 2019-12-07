from environment import Environment
from state_generator import State_generator
import time
import numpy as np

import rospy
from std_msgs.msg import Header
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose


class Vrep(Environment):
    def __init__(self, **kwargs):
        super().__init__()

        ### ------  VREP RESET  ------ ###
        self.reset_signal = False
        self.reset_pub = rospy.Publisher("/trpo/reset",Bool,queue_size=1)
        self.reset_sub = rospy.Subscriber("/simulation/reset",Bool,self.__reset_CB,queue_size=1)\

        ### ------  STATE GENERATION  ------ ###
        # Data Subscriber
        self.rs_image_sub = rospy.Subscriber("/vrep/depth_image",Image,self.__depth_CB, queue_size=1)
        self.jointstate_sub = rospy.Subscriber("/jaco/joint_states",JointState,self.__jointstate_CB, queue_size=1)
        self.pressure_sub = rospy.Subscriber("/vrep/pressure_data",Float32MultiArray,self.__pressure_CB, queue_size=1)

        # Data trigger switch
        self.depth_trigger = False
        self.joint_trigger = False
        self.pressure_trigger = False

        # Data Buffer
        self.image_buffersize = 5
        self.image_buff = []
        self.joint_buffersize = 30
        self.joint_state = []
        self.pressure_buffersize = 30
        self.pressure_state = []
        self.data_buff = []
        self.data_buff_temp = [0,0,0]

        # State Generator
        self.state_gen = State_generator(kwargs['sess'])

        ### ------  ACTION ------ ###
        # Action Publisher
        self.pose_pub = rospy.Publisher("/RL_controller/pub_pose",Pose,queue_size=10)

        ### ------  ENV PARAMETERS ------ ###
        self.create_environment(**kwargs)
        self.action_space = np.zeros(6) #x,y,z,r,p,y in Cartesian
        self.action_space_max = 0
        self.state_space = np.zeros(self.state_gen.state_size)


    def create_environment(self, **kwargs):
        #TODO:
        #if verp_jaco_bring has not been set, call.
        #else: Ignore and reset the environment once.
        self.reset_environment()
        pass

    def __reset_CB(self,msg):
        self.reset_signal = msg.data

    def reset_environment(self):
        #Publish trpo/reset_signal as std_msgs bool
        reset = Bool()
        reset.data = True
        self.reset_pub.publish(reset)
        #Wait till the simulation has reset    
        while True:
            if self.reset_signal:
                self.reset_signal = False
                break
        #Get current state from State generator
        self.current_state = self.__get_state()
        return self.current_state

    def get_state_shape(self):
        return self.state_space.shape

    def get_num_action(self):
        return self.action_space.shape[0]

    def get_action_bound(self):
        return self.action_space_max
    
    def step(self,action):
        time_required = self.__vrep_action_send(action) #action @ t
        # action time should be taken account into the next state transition or somewhere
        next_state, terminal_signal = self.__get_state() #state @ t+t()
        reward = [] #TODO
        if terminal_signal: #TODO
            terminal = True
        return np.reshape(next_state, [-1,]), reward, terminal

    def __vrep_action_send(self,action):
        #TODO: 
        # send action to vrep api and wait for the end signal
        # SHOULD return time required to finish the action
        # return time_required
        raise NotImplementedError

    def __depth_CB(self,msg):
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

    def __jointstate_CB(self,msg):
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

    def __pressure_CB(self,msg):
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

    def __get_state(self):
        return self.state_gen.generate(self.data_buff)