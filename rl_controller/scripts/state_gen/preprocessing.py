import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import time
import state_gen_util as state_gen_util
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from tqdm import tqdm
import rospkg
import datetime
import argparse

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

vrep_jaco_data_path = "../../../vrep_jaco_data"

""" Image data """
# data = [0,0,0,0]
background = np.load(vrep_jaco_data_path+"/data/data/background.npy",allow_pickle=True)
# print(background[0])
# plt.imshow(background[0],cmap='gray')
# plt.show()
# plt.imshow(background[1],cmap='gray')
# plt.show()
# plt.imshow(background[2],cmap='gray')
# plt.show()
# plt.imshow(background[3],cmap='gray')
# plt.show()
# exit()
data = [0]
data[0] = np.load(vrep_jaco_data_path+"/data/data/cam0_0.npy",allow_pickle=True)
data[0] = data[0][0:10]
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
            # np.where(cam[num]>5000*filter_rate,0,cam[num])
            img = cam[num]/5000
            img = 1-img
            img = np.reshape(img-background[idx],[480,640,1])
            test_tmp.append(img)
        else:
            # np.where(cam[num]>5000*filter_rate,0,cam[num])
            img = cam[num]/5000
            img = 1-img
            img = np.reshape(img-background[idx],[480,640,1])
            train_tmp.append(img)
    train_x.append(train_tmp)   
    test_x.append(test_tmp)   

train_x = np.array(train_x)
test_x = np.array(test_x)

input1 = train_x[0]

num = int(input('num'))
for _ in range(num):
    n = int(input('im_num'))
    plt.imshow(input1[n])
    plt.show()