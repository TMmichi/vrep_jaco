import os
import time
import state_gen_util as state_gen_util
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from tqdm import tqdm
import rospkg

###############################################
##### tensorflow setting gpu and ros path #####
###############################################

ros_path = rospkg.RosPack()

fig = plt.figure()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'


#############################
##### preparing dataset #####
#############################

""" Image data """
data = [0,0,0,0]
vrep_jaco_data_path = "../../../vrep_jaco_data"
data[0] = np.load(vrep_jaco_data_path+"/data/dummy_data.npy",allow_pickle=True)
data[1] = np.load(vrep_jaco_data_path+"/data/dummy_data.npy",allow_pickle=True)
data[2] = np.load(vrep_jaco_data_path+"/data/dummy_data.npy",allow_pickle=True)
data[3] = np.load(vrep_jaco_data_path+"/data/dummy_data.npy",allow_pickle=True)
data = np.array(data)

train_x = []
test_x = []

for num in range(data[0].shape[0]):
    train_tmp = []
    test_tmp = []
    if num%10==0:
        for idx, cam in enumerate(data):
            img = cam[num]/5000
            img = np.reshape(img,[480,640,1])
            test_tmp.append(img)
        test_x.append(test_tmp)
    else:
        for idx, cam in enumerate(data):
            img = cam[num]/5000
            img = np.reshape(img,[480,640,1])
            train_tmp.append(img)
        train_x.append(train_tmp)
train_x = np.array(train_x)
test_x = np.array(test_x)


####################
##### training #####
####################
""" training """
# train
epochs = 100

train = True
if train:
    """ build graph """
    autoencoder = state_gen_util.Autoencoder(debug=False)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    autoencoder.compile(optimizer=optimizer,
                        loss=autoencoder.compute_loss)
    autoencoder.fit(train_x,train_x,batch_size=1,epochs=100)
    autoencoder.save_weights('weights/mvae_autoencoder_weights')
else:
    autoencoder_load = state_gen_util.Autoencoder(debug=False)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    autoencoder_load.compile(optimizer=optimizer,
                        loss = autoencoder_load.compute_loss)
    autoencoder_load.load_weights('weights/mvae_autoencoder_weights')
    autoencoder_load.summary()
    
    def onChange(hey):
        pass

    eval_state_1 = np.zeros((1,32))
    sampled_pic_1 = autoencoder_load.sample(1,eval_state_1).numpy()
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
        
        sampled_pic_1 = autoencoder_load.sample(1,eval_state_1).numpy()
        sampled_pic_1 = np.tile(sampled_pic_1[0],(1,1,3))

    '''
    eval_pic = autoencoder_load.predict(test_x)        
    eval_pic = np.tile(eval_pic[0],(1,1,3))
    gt_pic = np.tile(test_x[0],(1,1,3))
    
    eval_states = autoencoder_load.state(test_x).numpy()
    eval_state_02 = eval_states[2] - 0.2
    eval_state_02 = np.reshape(eval_state_02,[1,eval_state_02.shape[0]])
    sampled_pic_02 = autoencoder_load.sample(1,eval_state_02).numpy()
    sampled_pic_02 = np.tile(sampled_pic_02[0],(1,1,3))
    gt_pic_02 = np.tile(test_x[2],(1,1,3))

    eval_state_05 = eval_states[5] - 0.5
    eval_state_05 = np.reshape(eval_state_05,[1,eval_state_05.shape[0]])
    sampled_pic_05 = autoencoder_load.sample(1,eval_state_05).numpy()
    sampled_pic_05 = np.tile(sampled_pic_05[0],(1,1,3))
    gt_pic_05 = np.tile(test_x[5],(1,1,3))

    idx = 11
    eval_state_1 = eval_states[idx]
    
    for i in range(len(eval_state_1)):
        #eval_state_1[i] += 2*np.random.randn(1)
        eval_state_1[i]  = 0.1*i-4
    eval_state_1 = np.reshape(eval_state_1,[1,eval_state_1.shape[0]])
    sampled_pic_1 = autoencoder_load.sample(1,eval_state_1).numpy()
    sampled_pic_1 = np.tile(sampled_pic_1[0],(1,1,3))
    gt_pic_1 = np.tile(test_x[idx],(1,1,3))
    
    ax11 = fig.add_subplot(4,2,1)
    ax11.imshow(eval_pic)
    ax12 = fig.add_subplot(4,2,2)
    ax12.imshow(gt_pic)
    ax21 = fig.add_subplot(4,2,3)
    ax21.imshow(sampled_pic_02)
    ax22 = fig.add_subplot(4,2,4)
    ax22.imshow(gt_pic_02)
    ax31 = fig.add_subplot(4,2,5)
    ax31.imshow(sampled_pic_05)
    ax32 = fig.add_subplot(4,2,6)
    ax32.imshow(gt_pic_05)
    ax41 = fig.add_subplot(4,2,7)
    ax41.imshow(sampled_pic_1)
    ax42 = fig.add_subplot(4,2,8)
    ax42.imshow(gt_pic_1)
    #plt.show()'''

