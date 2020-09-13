import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime
import os
import rospkg
import argparse
from sklearn.decomposition import PCA


##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
''' data preparing section '''
data = [0,0,0,0]
vrep_jaco_data_path = "../../../vrep_jaco_data"
data[0] = np.load(vrep_jaco_data_path+"/data/data/dummy_data1.npy",allow_pickle=True)
data[1] = np.load(vrep_jaco_data_path+"/data/data/dummy_data2.npy",allow_pickle=True)
data[2] = np.load(vrep_jaco_data_path+"/data/data/dummy_data3.npy",allow_pickle=True)
data[3] = np.load(vrep_jaco_data_path+"/data/data/dummy_data4.npy",allow_pickle=True)
data = np.array(data)

train_x = []
test_x = []

for idx, cam in enumerate(data):
    train_tmp = []
    test_tmp = []
    for num in range(data[0].shape[0]):
        if num%10==0:
            # cam[num] = np.where(cam[num]>5000*filter_rate,5000,cam[num])
            img = cam[num]/5000
            img = 1-img
            img = np.reshape(img,[480,640,1])
            test_tmp.append(img)
        else:
            # cam[num] = np.where(cam[num]>5000*filter_rate,5000,cam[num])
            img = cam[num]/5000
            img = 1-img
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

tmp = np.reshape(input1,(-1,480*640*1))
pca = PCA(n_components=1)
print('pca start')
pca.fit(tmp)
print('pca end')
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

tmp_pca = pca.transform(tmp)
# tmp_pca =
tmp_pca2 = (tmp - pca.mean_).dot(pca.components_.T)

tmp_projected = pca.inverse_transform(tmp_pca)
tmp_projected2 = tmp_pca.dot(pca.components_) + pca.mean_

loss = ((tmp - tmp_projected) ** 2).mean()
print(loss)

im_origin = np.reshape(tmp,(-1,480,640))
im_projected = np.reshape(tmp_projected,(-1,480,640))

itr_ = int(input('input your iter time'))

for _ in range(itr_):
    idx = int(input('input your number of im'))
    plt.imshow(im_origin[idx])
    plt.show()
    plt.imshow(im_projected[idx])
    plt.show()










