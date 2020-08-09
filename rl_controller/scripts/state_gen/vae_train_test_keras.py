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

###############################################
##### tensorflow setting gpu and ros path #####
###############################################
ros_path = rospkg.RosPack()
fig = plt.figure()
# with tf.device('/device:GPU:0'):

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


#############################
##### preparing dataset #####
#############################

""" Image data """
data = [0,0,0,0]
vrep_jaco_data_path = "../../../vrep_jaco_data"
data[0] = np.load(vrep_jaco_data_path+"/data/data_new1.npy",allow_pickle=True)
print(data[0].shape)
data[1] = np.load(vrep_jaco_data_path+"/data/data_new2.npy",allow_pickle=True)
print(data[1].shape)
data[2] = np.load(vrep_jaco_data_path+"/data/data_new3.npy",allow_pickle=True)
print(data[2].shape)
data[3] = np.load(vrep_jaco_data_path+"/data/data_new4.npy",allow_pickle=True)
print(data[3].shape)
data = np.array(data)

# data1 = np.load(vrep_jaco_data_path+"/data/dummy_data0602_2.npy",allow_pickle=True)
# data2 = np.load(vrep_jaco_data_path+"/data/dummy_data0602_3.npy",allow_pickle=True)
# data3 = np.load(vrep_jaco_data_path+"/data/dummy_data0602_6.npy",allow_pickle=True)
# data = np.array([data1,data2,data3])

train_x = []
test_x = []

for idx, cam in enumerate(data):
    train_tmp = []
    test_tmp = []
    for num in range(data[0].shape[0]):
        if num%10==0:
            img = cam[num]/5000
            img = np.reshape(img,[480,640,1])
            test_tmp.append(img)
        else:
            img = cam[num]/5000
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
# a = np.reshape(input1[0],(480,640))
# plt.imshow(a)
# plt.show()
# print('end?')
# quit()


####################
##### training #####
####################
""" training """
# train
epochs = 1000

train = False
# train = True

if train:
    """ build graph """
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    a = state_gen_util.Autoencoder(debug=False,training=False)
    autoencoder = a.autoencoder
    optimizer = tf.keras.optimizers.Adam(1e-5)
    autoencoder.compile(optimizer=optimizer,
                        loss=[a.kl_reconstruction_loss1,a.kl_reconstruction_loss2,a.kl_reconstruction_loss3,a.kl_reconstruction_loss4])
    autoencoder.load_weights('weights/mvae_autoencoder_weights')
    
    autoencoder.fit(x=[input1,input2,input3,input4],y=[input1,input2,input3,input4],batch_size=20,epochs=100,callbacks=[tensorboard_callback])
    autoencoder.save_weights('weights/mvae_autoencoder_weights_150')
else:
    a = state_gen_util.Autoencoder(debug=False,training=False)
    autoencoder_load = a.autoencoder
    optimizer = tf.keras.optimizers.Adam(1e-4)
    autoencoder_load.compile(optimizer=optimizer,
                        loss=[a.kl_reconstruction_loss1,a.kl_reconstruction_loss2,a.kl_reconstruction_loss3,a.kl_reconstruction_loss4])
    autoencoder_load.load_weights('weights/mvae_autoencoder_weights_150')
    
    def onChange(hey):
        pass



    ###############
    ### drawing ###
    ###############
    
    print('this1?')
    fig=plt.figure(figsize=(100, 60))

    

    haha1 = np.array([test_x[0][0]])
    haha2 = np.array([test_x[1][0]])
    haha3 = np.array([test_x[2][0]])
    haha4 = np.array([test_x[3][0]])

    haha1_ = np.array([train_x[0][0]])
    haha2_ = np.array([train_x[1][0]])
    haha3_ = np.array([train_x[2][0]])
    haha4_ = np.array([train_x[3][0]])

    z1 , _ = a.encoder.encoder_model1.predict([haha1_])
    z2 , _ = a.encoder.encoder_model2.predict([haha2_])
    z3 , _ = a.encoder.encoder_model3.predict([haha3_])
    z4 , _ = a.encoder.encoder_model4.predict([haha4_])

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
    print('this2?')

    rows = 2
    columns = 4
    for i in range(1, columns*rows +1):
        if i<5:
            fig.add_subplot(rows, columns, i)
            plt.imshow(haha[i-1])
            # plt.imshow(haha_[i-1])
            print('thi31?')

        else:
            fig.add_subplot(rows, columns, i)
            plt.imshow(pics[i-5])
    plt.show()
    print('this4?')

    quit()
    
    sampled_pic_1 = np.tile(sampled_pic_1[0],(1,1,3))

    window_name = 'latent vector to img'
    cv.namedWindow(window_name)
    for i in range(16):
        tkbar_name = 'idx:'+str(i*2)
        cv.createTrackbar(tkbar_name,window_name, -30, 30, onChange)
    
    while True:
        cv.imshow(window_name, sampled_pic_1)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        
        for i in range(int(len(eval_state_1[0])/2)):
            tkbar_name = 'idx:'+str(2*i)
            value = cv.getTrackbarPos(tkbar_name, window_name)/10
            eval_state_1[0][2*i] = value
            eval_state_1[0][2*i+1] = value
        
        sampled_pic_1 = a.sample(1,eval_state_1).numpy()
        sampled_pic_2 = a.sample(1,eval_state_2).numpy()
        sampled_pic_3 = a.sample(1,eval_state_3).numpy()
        sampled_pic_4 = a.sample(1,eval_state_4).numpy()

        sampled_pic_1 = np.tile(sampled_pic_1[0],(1,1,3))
        sampled_pic_2 = np.tile(sampled_pic_2[0],(1,1,3))
        sampled_pic_3 = np.tile(sampled_pic_3[0],(1,1,3))
        sampled_pic_4 = np.tile(sampled_pic_4[0],(1,1,3))
