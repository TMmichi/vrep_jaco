import tensorflow as tf
import numpy as np
import os
import vae_util

state_size = 10

class State_generator:
    def __init__(self,sess):
        self.state_size = state_size
        self.sess = sess
        self.inputs, self.output = self._build_graph()

    def generate(self,data_buff):
        #data_buff -> [data@timestep][data_type][data,timestamp]
        image_arm = data_buff[-1][0][0]/5000
        image_bed = data_buff[-1][1][0]/5000
        image_arm = np.reshape(image_arm,[1,image_arm.shape[0],image_arm.shape[1],1])
        image_bed = np.reshape(image_bed,[1,image_bed.shape[0],image_bed.shape[1],1])
        jointstate_data = np.array(data_buff[-1][2][0])
        pressure_data = np.array(data_buff[-1][3][0])
        inputs = dict({self.inputs[0]:image_arm, self.inputs[1]:image_bed, self.inputs[2]:1})
        inputs[self.inputs[3]]=jointstate_data
        inputs[self.inputs[4]]=pressure_data
        return self.sess.run((self.output),feed_dict=inputs)

    def _build_graph(self):
        image_arm_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 480,640,1], name='input_img_arm')
        image_bed_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 480,640,1], name='input_img_bed')
        #pressure_data = 
        #jointstate_data = 
        keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        mean_arm, stddev_arm = vae_util.CNN_Encoder(image_arm_input, z_dim=64, trainable=False)
        mean_bed, stddev_bed = vae_util.CNN_Encoder(image_bed_input, z_dim=64, trainable=False)
        output = []

        return [image_arm_input, image_bed_input, keep_prob,], output
        
