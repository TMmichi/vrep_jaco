import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import time
import state_gen_util as state_gen_util
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from tqdm import tqdm
import rospkg
import datetime
import argparse
import tensorwatch as tw

######################################
##### argparse for setting value #####
######################################
parser = argparse.ArgumentParser(description="epochs과 train여부를 입력해주세요")
parser.add_argument('--epochs',required=False,default=1000,help='epochs 수')
parser.add_argument('--train',required=False,default=True,help='train 여부')
parser.add_argument('--drawing',required=False,default=False,help='drawing 여부')
parser.add_argument('--filter',required=False,default=1.0,help='filter 비율')
parser.add_argument('--load',required=False,default=False,help='weight load 여부')
parser.add_argument('--weight_name',required=False,default='weights/mvae_autoencoder_weights',help='weight load 이름')
parser.add_argument('--only_vae',required=False,default=False,help='vae로 학습')

args = parser.parse_args()
epochs = args.epochs
train = args.train
drawing = args.drawing
filter_rate = args.filter
load = args.load
weight_name = args.weight_name
only_vae = args.only_vae

###############################################
##### tensorflow setting gpu and ros path #####
###############################################
fig = plt.figure()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'


#############################
##### preparing dataset #####
#############################

print("###########################################################################")
print("##########################     preprocessing     ##########################")
print("###########################################################################")
print("                                                                           ")
print("                                                                           ")
print("                                                                           ")

""" Image data """
# data = [0,0,0,0]
data = [0]
vrep_jaco_data_path = "../../../vrep_jaco_data"
data[0] = np.load(vrep_jaco_data_path+"/data/data/dummy_data1.npy",allow_pickle=True)
# data[1] = np.load(vrep_jaco_data_path+"/data/data/dummy_data2.npy",allow_pickle=True)
# data[2] = np.load(vrep_jaco_data_path+"/data/data/dummy_data3.npy",allow_pickle=True)
# data[3] = np.load(vrep_jaco_data_path+"/data/data/dummy_data4.npy",allow_pickle=True)
data = np.array(data)

train_x = []
test_x = []

for idx, cam in enumerate(data):
    train_tmp = []
    test_tmp = []
    for num in range(data[0].shape[0]):
        if num%10==0:
            np.where(cam[num]>5000*filter_rate,0,cam[num])
            img = cam[num]/5000
            img = np.reshape(img,[480,640,1])
            test_tmp.append(img)
        else:
            np.where(cam[num]>5000*filter_rate,0,cam[num])
            img = cam[num]/5000
            img = np.reshape(img,[480,640,1])
            train_tmp.append(img)
    train_x.append(train_tmp)   
    test_x.append(test_tmp)   

train_x = np.array(train_x)
test_x = np.array(test_x)

input1 = train_x[0]
# input2 = train_x[1]
# input3 = train_x[2]
# input4 = train_x[3]


####################
##### training #####
####################
print("###########################################################################")
print("##########################   train    learning   ##########################")
print("###########################################################################")
print("                                                                           ")
print("                                                                           ")
print("                                                                           ")


""" training """
if train:
    if only_vae:
        """ build graph """
        
        """ tensorboard setting """
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_images=True)
        # latent_z_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[1].get_weights()))

        """ model setting """
        input_ = tf.keras.Input(shape=(480,640,1), name='vae_input')
        model_ = state_gen_util.VAE_model(debug=True,num_conv_blocks=5,num_deconv_blocks=5)
        result_ = model_(input_)
        model = tf.keras.Model(input_,result_)
        optimizer = tf.keras.optimizers.Adam(1e-3)
        model.compile(optimizer=optimizer, loss=model_.vae_loss)
        if load == True:
            model.load_weights(weight_name)
        print(model.summary())

        """ model train """
        # model.fit(x=input1,y=input1,batch_size=2,epochs=100,callbacks=[tensorboard_callback,latent_z_callback])
        model.fit(x=input1,y=input1,batch_size=2,epochs=100,callbacks=[tensorboard_callback])
        model.save_weights('weights/only_vae_autoencoder_weights')
    
    else:
        """ build graph """
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        a = state_gen_util.Autoencoder(debug=False,training=False)
        autoencoder = a.autoencoder
        optimizer = tf.keras.optimizers.Adam(1e-3)
        autoencoder.compile(optimizer=optimizer,
                            loss=[a.kl_reconstruction_loss1,a.kl_reconstruction_loss2,a.kl_reconstruction_loss3,a.kl_reconstruction_loss4])
        if load == True:
            autoencoder.load_weights(weight_name)
        print(autoencoder.summary())
        autoencoder.fit(x=[input1,input2,input3,input4],y=[input1,input2,input3,input4],batch_size=2,epochs=100,callbacks=[tensorboard_callback])
        autoencoder.save_weights('weights/mvae_autoencoder_weights')
else:
    a = state_gen_util.Autoencoder(debug=False,training=False)
    autoencoder_load = a.autoencoder
    optimizer = tf.keras.optimizers.Adam(1e-4)
    autoencoder_load.compile(optimizer=optimizer,
                        loss=[a.kl_reconstruction_loss1,a.kl_reconstruction_loss2,a.kl_reconstruction_loss3,a.kl_reconstruction_loss4])
    autoencoder_load.load_weights('weights/mvae_autoencoder_weights')

    ###############
    ### drawing ###
    ###############
    if drawing==True:
        fig=plt.figure(figsize=(640, 480))

        haha1 = np.array([test_x[0][0]])
        haha2 = np.array([test_x[1][0]])
        haha3 = np.array([test_x[2][0]])
        haha4 = np.array([test_x[3][0]])

        haha1_ = np.array([train_x[0][0]])
        haha2_ = np.array([train_x[1][0]])
        haha3_ = np.array([train_x[2][0]])
        haha4_ = np.array([train_x[3][0]])

        z1 , _ = a.encoder.encoder_model1.predict([haha1])
        z2 , _ = a.encoder.encoder_model2.predict([haha2])
        z3 , _ = a.encoder.encoder_model3.predict([haha3])
        z4 , _ = a.encoder.encoder_model4.predict([haha4])

        sampled_pic1,sampled_pic2,sampled_pic3,sampled_pic4 = a.sample(1,[z1,z2,z3,z4])
        sampled_pic1 = np.reshape(sampled_pic1,(480,640))
        sampled_pic2 = np.reshape(sampled_pic2,(480,640))
        sampled_pic3 = np.reshape(sampled_pic3,(480,640))
        sampled_pic4 = np.reshape(sampled_pic4,(480,640))

        haha1 = np.reshape(haha1,(480,640))
        haha2 = np.reshape(haha2,(480,640))
        haha3 = np.reshape(haha3,(480,640))
        haha4 = np.reshape(haha4,(480,640))

        haha1_ = np.reshape(haha1_,(480,640))
        haha2_ = np.reshape(haha2_,(480,640))
        haha3_ = np.reshape(haha3_,(480,640))
        haha4_ = np.reshape(haha4_,(480,640))

        haha = [haha1,haha2,haha3,haha4]
        haha_ = [haha1_,haha2_,haha3_,haha4_]
        pics = [sampled_pic1,sampled_pic2,sampled_pic3,sampled_pic4]
        # plt.imshow(np.reshape(haha1,(480,640)))
        # plt.imshow(sampled_pic)

        rows = 2
        columns = 4
        for i in range(1, columns*rows +1):
            if i<5:
                fig.add_subplot(rows, columns, i)
                plt.imshow(haha[i-1])
                # plt.imshow(haha_[i-1])
            else:
                fig.add_subplot(rows, columns, i)
                plt.imshow(pics[i-5])
        plt.show()


