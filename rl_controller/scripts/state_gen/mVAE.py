import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime
import os
import rospkg
import argparse



##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
'''
******************************************
|| epoch       || 수                         
|| train       || 여부                       
|| filter      || depth거리 어느정도 자를지 여부
|| drawing     || 결과 plot                   
|| load        || trained weight load여부     
|| weight_name || weight경로                  
|| only_vae    || 단순 vae만 사용             
******************************************
'''
parser = argparse.ArgumentParser(description="epochs과 train여부를 입력해주세요")
parser.add_argument('--epochs',required=False,default=200,help='epochs 수')
parser.add_argument('--train',required=False,default=True,help='train 여부')
parser.add_argument('--drawing',required=False,default=False,help='drawing 여부')
parser.add_argument('--filter',required=False,default=0.5,help='filter 비율')
parser.add_argument('--load',required=False,default=False,help='weight load 여부')
parser.add_argument('--weight_name',required=False,default='weights/mvae_autoencoder_weights',help='weight load 이름')
parser.add_argument('--only_vae',required=False,default=False,help='vae로 학습')

args = parser.parse_args()
epochs = int(args.epochs)
train = bool(int(args.train))
drawing = bool(int(args.drawing))
filter_rate = float(args.filter)
load = bool(int(args.load))
weight_name = args.weight_name
only_vae = bool(int(args.only_vae))
''' debugging '''
# print(epochs, train, drawing, filter_rate, load, weight_name, only_vae)
# exit()



##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
''' data preparing section '''
data = [0,0,0,0]
vrep_jaco_data_path = "../../../vrep_jaco_data"
background = np.load(vrep_jaco_data_path+"/data/data/background.npy",allow_pickle=True)
# data[0] = np.load(vrep_jaco_data_path+"/data/data/dummy_data1.npy",allow_pickle=True)
# data[1] = np.load(vrep_jaco_data_path+"/data/data/dummy_data2.npy",allow_pickle=True)
# data[2] = np.load(vrep_jaco_data_path+"/data/data/dummy_data3.npy",allow_pickle=True)
# data[3] = np.load(vrep_jaco_data_path+"/data/data/dummy_data4.npy",allow_pickle=True)
data[0] = np.load(vrep_jaco_data_path+"/data/data/cam0_0_0.npy",allow_pickle=True)
data[0] = data[0][:300]
data[1] = np.load(vrep_jaco_data_path+"/data/data/cam1_0_0.npy",allow_pickle=True)
data[1] = data[1][:300]
data[2] = np.load(vrep_jaco_data_path+"/data/data/cam2_0_0.npy",allow_pickle=True)
data[2] = data[2][:300]
data[3] = np.load(vrep_jaco_data_path+"/data/data/cam3_0_0.npy",allow_pickle=True)
data[3] = data[3][:300]
data = np.array(data)
# background = 1 - background
# plt.imshow(np.reshape(background[0],(480,640)),cmap='gray')
# plt.show()
# plt.imshow(np.reshape(data[0][0]/5000,(480,640)),cmap='gray')
# plt.show()
# plt.imshow(np.reshape((data[0][0]/5000 - background[0]),(480,640)),cmap='gray')
# plt.show()
# exit()
train_x = []
test_x = []

for idx, cam in enumerate(data):
    train_tmp = []
    test_tmp = []
    for num in range(data[0].shape[0]):
        if num%10==0:
            # cam[num] = np.where(cam[num]>5000*filter_rate,5000,cam[num])
            img = cam[num]/5000
            # print(img)
            # print(background[idx])
            # print(background[0])
            img = 1-img
            img = img-background[idx]
            img = np.where(img<0,0,img)
            img = np.reshape(img,[480,640,1])
            test_tmp.append(img)
        else:
            # cam[num] = np.where(cam[num]>5000*filter_rate,5000,cam[num])
            img = cam[num]/5000
            img = 1-img
            img = img-background[idx]
            # print(img)
            # exit()
            img = np.where(img<0,0,img)
            img = np.reshape(img,[480,640,1])
            train_tmp.append(img)
    train_x.append(train_tmp)   
    test_x.append(test_tmp)   

train_x = np.array(train_x)
test_x = np.array(test_x)

input1 = train_x[0]
input2 = train_x[1]
input3 = train_x[2]
input4 = train_x[3]

test_input1 = test_x[0]
test_input2 = test_x[1]
test_input3 = test_x[2]
test_input4 = test_x[3]


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_images=True)
# latent_z_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[1].get_weights()))
# print(input1[0])
# plt.imshow(np.reshape(input1[0],(480,640)),cmap='gray')
# plt.show()


##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
''' preparing model '''

# latent dim
latent_dim = 512

# sampling function
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class mVAE(keras.Model):
    def __init__(self, **kwargs):
        super(mVAE, self).__init__()
        self._build()
    
    def _build(self):
        #####################################
        ########## setting encoder ##########
        #####################################
        encoder_inputs_1 = keras.Input(shape=(480, 640, 1))
        x_1 = layers.Conv2D(16, 8, activation="swish", strides=2, padding="same")(encoder_inputs_1)
        x_1 = layers.Dropout(0.2)(x_1)
        x_1 = layers.Conv2D(32, 4, activation="swish", strides=2, padding="same")(x_1)
        x_1 = layers.Dropout(0.2)(x_1)
        x_1 = layers.Conv2D(64, 4, activation="swish", strides=2, padding="same")(x_1)
        x_1 = layers.Dropout(0.2)(x_1)
        x_1 = layers.Conv2D(128, 3, activation="swish", strides=2, padding="same")(x_1)
        x_1 = layers.Dropout(0.2)(x_1)
        x_1 = layers.Conv2D(256, 3, activation="swish", strides=2, padding="same")(x_1)
        x_1 = layers.Dropout(0.2)(x_1)
        x_1 = layers.Flatten()(x_1)
        x_1 = layers.Dense(512, activation="swish")(x_1)
        z1 = layers.Dense(512, name="z1")(x_1)

        encoder_inputs_2 = keras.Input(shape=(480, 640, 1))
        x_2 = layers.Conv2D(16, 8, activation="swish", strides=2, padding="same")(encoder_inputs_2)
        x_2 = layers.Dropout(0.2)(x_2)
        x_2 = layers.Conv2D(32, 4, activation="swish", strides=2, padding="same")(x_2)
        x_2 = layers.Dropout(0.2)(x_2)
        x_2 = layers.Conv2D(64, 4, activation="swish", strides=2, padding="same")(x_2)
        x_2 = layers.Dropout(0.2)(x_2)
        x_2 = layers.Conv2D(128, 3, activation="swish", strides=2, padding="same")(x_2)
        x_2 = layers.Dropout(0.2)(x_2)
        x_2 = layers.Conv2D(256, 3, activation="swish", strides=2, padding="same")(x_2)
        x_2 = layers.Dropout(0.2)(x_2)
        x_2 = layers.Flatten()(x_2)
        x_2 = layers.Dense(512, activation="swish")(x_2)
        z2 = layers.Dense(512, name="z2")(x_2)

        encoder_inputs_3 = keras.Input(shape=(480, 640, 1))
        x_3 = layers.Conv2D(16, 8, activation="swish", strides=2, padding="same")(encoder_inputs_3)
        x_3 = layers.Dropout(0.2)(x_3)
        x_3 = layers.Conv2D(32, 4, activation="swish", strides=2, padding="same")(x_3)
        x_3 = layers.Dropout(0.2)(x_3)
        x_3 = layers.Conv2D(64, 4, activation="swish", strides=2, padding="same")(x_3)
        x_3 = layers.Dropout(0.2)(x_3)
        x_3 = layers.Conv2D(128, 3, activation="swish", strides=2, padding="same")(x_3)
        x_3 = layers.Dropout(0.2)(x_3)
        x_3 = layers.Conv2D(256, 3, activation="swish", strides=2, padding="same")(x_3)
        x_3 = layers.Dropout(0.2)(x_3)
        x_3 = layers.Flatten()(x_3)
        x_3 = layers.Dense(512, activation="swish")(x_3)
        z3 = layers.Dense(512, name="z3")(x_3)

        encoder_inputs_4 = keras.Input(shape=(480, 640, 1))
        x_4 = layers.Conv2D(16, 8, activation="swish", strides=2, padding="same")(encoder_inputs_4)
        x_4 = layers.Dropout(0.2)(x_4)
        x_4 = layers.Conv2D(32, 4, activation="swish", strides=2, padding="same")(x_4)
        x_4 = layers.Dropout(0.2)(x_4)
        x_4 = layers.Conv2D(64, 4, activation="swish", strides=2, padding="same")(x_4)
        x_4 = layers.Dropout(0.2)(x_4)
        x_4 = layers.Conv2D(128, 3, activation="swish", strides=2, padding="same")(x_4)
        x_4 = layers.Dropout(0.2)(x_4)
        x_4 = layers.Conv2D(256, 3, activation="swish", strides=2, padding="same")(x_4)
        x_4 = layers.Dropout(0.2)(x_4)
        x_4 = layers.Flatten()(x_4)
        x_4 = layers.Dense(512, activation="swish")(x_4)
        z4 = layers.Dense(512, name="z4")(x_4)

        z_ = tf.concat([z1,z2,z3,z4],axis=1)

        z_ = layers.Dense(512, activation="swish")(z_)
        z_ = layers.Dropout(0.2)(z_)

        z_mean = layers.Dense(latent_dim, name="z_mean")(z_)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(z_)

        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model([encoder_inputs_1,encoder_inputs_2,encoder_inputs_3,encoder_inputs_4], [z_mean, z_log_var, z], name="encoder")
        
        #####################################
        ########## setting decoder ##########
        #####################################
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(512, activation="swish")(latent_inputs)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation="swish")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation="swish")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation="swish")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(4*15*20*128, activation="swish")(x)

        x1, x2, x3, x4 = tf.split(x, num_or_size_splits=4, axis=1, num=None, name='split')

        x1 = layers.Reshape((15, 20, 128))(x1)
        x2 = layers.Reshape((15, 20, 128))(x2)
        x3 = layers.Reshape((15, 20, 128))(x3)
        x4 = layers.Reshape((15, 20, 128))(x4)

        x1 = layers.Conv2DTranspose(256, 3, activation="swish", strides=2, padding="same")(x1)
        x1 = layers.Conv2DTranspose(128, 3, activation="swish", strides=2, padding="same")(x1)
        x1 = layers.Conv2DTranspose(64, 4, activation="swish", strides=2, padding="same")(x1)
        x1 = layers.Conv2DTranspose(32, 4, activation="swish", strides=2, padding="same")(x1)
        x1 = layers.Conv2DTranspose(16, 4, activation="swish", strides=2, padding="same")(x1)
        decoder_outputs_1 = layers.Conv2DTranspose(1, 8, activation="sigmoid", padding="same", name="final1")(x1)

        x2 = layers.Conv2DTranspose(256, 3, activation="swish", strides=2, padding="same")(x2)
        x2 = layers.Conv2DTranspose(128, 3, activation="swish", strides=2, padding="same")(x2)
        x2 = layers.Conv2DTranspose(64, 4, activation="swish", strides=2, padding="same")(x2)
        x2 = layers.Conv2DTranspose(32, 4, activation="swish", strides=2, padding="same")(x2)
        x2 = layers.Conv2DTranspose(16, 4, activation="swish", strides=2, padding="same")(x2)
        decoder_outputs_2 = layers.Conv2DTranspose(1, 8, activation="sigmoid", padding="same", name="final2")(x2)

        x3 = layers.Conv2DTranspose(256, 3, activation="swish", strides=2, padding="same")(x3)
        x3 = layers.Conv2DTranspose(128, 3, activation="swish", strides=2, padding="same")(x3)
        x3 = layers.Conv2DTranspose(64, 4, activation="swish", strides=2, padding="same")(x3)
        x3 = layers.Conv2DTranspose(32, 4, activation="swish", strides=2, padding="same")(x3)
        x3 = layers.Conv2DTranspose(16, 4, activation="swish", strides=2, padding="same")(x3)
        decoder_outputs_3 = layers.Conv2DTranspose(1, 8, activation="sigmoid", padding="same", name="final3")(x3)

        x4 = layers.Conv2DTranspose(256, 3, activation="swish", strides=2, padding="same")(x4)
        x4 = layers.Conv2DTranspose(128, 3, activation="swish", strides=2, padding="same")(x4)
        x4 = layers.Conv2DTranspose(64, 4, activation="swish", strides=2, padding="same")(x4)
        x4 = layers.Conv2DTranspose(32, 4, activation="swish", strides=2, padding="same")(x4)
        x4 = layers.Conv2DTranspose(16, 4, activation="swish", strides=2, padding="same")(x4)
        decoder_outputs_4 = layers.Conv2DTranspose(1, 8, activation="sigmoid", padding="same", name="final4")(x4)

        self.decoder = keras.Model(latent_inputs, [decoder_outputs_1,decoder_outputs_2,decoder_outputs_3,decoder_outputs_4], name="decoder")
    
    def show_model_structure(self):
        print('encoder structure')
        print(self.encoder.summary())
        print('decoder structure')
        print(self.decoder.summary())

    def plot_latent(self):
        figsize1 = 480
        figsize2 = 640

        iter_ = int(input('set number of iteraion: '))
        
        for _ in range(iter_):
            k = int(input('set number of test image: '))
            ins1 = np.reshape(test_input1[k],(1,480,640,1))
            ins2 = np.reshape(test_input2[k],(1,480,640,1))
            ins3 = np.reshape(test_input3[k],(1,480,640,1))
            ins4 = np.reshape(test_input4[k],(1,480,640,1))
            
            z1, z2, z_ = self.encoder.predict([ins1,ins2,ins3,ins4])
            print(z_)
            figure1, figure2, figure3, figure4 = self.decoder.predict(z_)
            figure1 = np.reshape(figure1,(480,640))
            figure2 = np.reshape(figure2,(480,640))
            figure3 = np.reshape(figure3,(480,640))
            figure4 = np.reshape(figure4,(480,640))
            
            figure1_origin = np.reshape(test_input1[k],(480,640))
            figure2_origin = np.reshape(test_input2[k],(480,640))
            figure3_origin = np.reshape(test_input3[k],(480,640))
            figure4_origin = np.reshape(test_input4[k],(480,640))
            
            origin = [figure1_origin,figure2_origin,figure3_origin,figure4_origin]
            result = [figure1,figure2,figure3,figure4]
            
            fig = plt.figure(figsize=(figsize1, figsize2))
            rows = 2
            columns = 4
            for i in range(1, columns*rows +1):
                if i<5:
                    fig.add_subplot(rows, columns, i)
                    plt.imshow(origin[i-1])
                else:
                    fig.add_subplot(rows, columns, i)
                    plt.imshow(result[i-5])
            plt.show()

    def train_step(self, data):
        if isinstance(data, tuple):
            data1 = data[0][0]
            data2 = data[0][1]
            data3 = data[0][2]
            data4 = data[0][3]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([data1,data2,data3,data4])
            reconstruction1, reconstruction2, reconstruction3, reconstruction4 = self.decoder(z)
            
            reconstruction_loss1 = tf.reduce_mean(
                keras.losses.binary_crossentropy(data1, reconstruction1)
            )
            reconstruction_loss2 = tf.reduce_mean(
                keras.losses.binary_crossentropy(data2, reconstruction2)
            )
            reconstruction_loss3 = tf.reduce_mean(
                keras.losses.binary_crossentropy(data3, reconstruction3)
            )
            reconstruction_loss4 = tf.reduce_mean(
                keras.losses.binary_crossentropy(data4, reconstruction4)
            )
            
            reconstruction_loss1 *= 480 * 640 # 1 * 1 #
            reconstruction_loss2 *= 480 * 640 # 1 * 1 #
            reconstruction_loss3 *= 480 * 640 # 1 * 1 #
            reconstruction_loss4 *= 480 * 640 # 1 * 1 #
            reconstruction_loss = reconstruction_loss1 + reconstruction_loss2 + reconstruction_loss3 + reconstruction_loss4

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
               
        return {
            "loss": total_loss,
            "reconstruction_loss1": reconstruction_loss1,
            "reconstruction_loss2": reconstruction_loss2,
            "reconstruction_loss3": reconstruction_loss3,
            "reconstruction_loss4": reconstruction_loss4,
            "kl_loss": kl_loss,
        }


if __name__=='__main__':
    vae = mVAE()
    vae.compile(optimizer=keras.optimizers.Nadam(1e-4))
    # vae.compile(optimizer=keras.optimizers.Adam(1e-4))
    vae.show_model_structure()
    
    if train==True:
        if load==True:
            vae.load_weights(weight_name)
        print('here?')
        vae.fit([input1,input2,input3,input4], epochs=epochs, batch_size=4, callbacks=[tensorboard_callback])
        save_weight_name_encoder = "weights/en_mVAE_weights_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_weight_name_decoder = "weights/de_mVAE_weights_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # vae.encoder.save_weights(save_weight_name_encoder,save_format=)
        # vae.decoder.save_weights(save_weight_name_decoder)
        vae.save_weights('./weights/my_model')
    
    if drawing==True:
        if load==True:
            vae.load_weights('./weights/my_model.data-00000-of-00001')
            # vae.encoder.load_weights(weight_name,by_name=True)
            # vae.decoder.load_weights(weight_name,by_name=True)
        vae.plot_latent()