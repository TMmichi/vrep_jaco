import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
'''
num = input("data의 총 개수를 입력해주세요 : ")
num = int(num)
idx=0
data0=[]
data1=[]
data2=[]
data3=[]
for i in tqdm(range(num)):
    if i%150==0 and i!=0:
        np.save("cam0_"+str(idx),data0)
        np.save("cam1_"+str(idx),data1)
        np.save("cam2_"+str(idx),data2)
        np.save("cam3_"+str(idx),data3)
        idx+=1
        data0=[]
        data1=[]
        data2=[]
        data3=[]
        a = np.load("./data_before/"+str(i)+".npy",allow_pickle=True)
        data0.append(a[0][0])
        data1.append(a[1][0])
        data2.append(a[2][0])
        data3.append(a[3][0])
    else:
        a = np.load("./data_before/"+str(i)+".npy",allow_pickle=True)
        data0.append(a[0][0])
        data1.append(a[1][0])
        data2.append(a[2][0])
        data3.append(a[3][0])

for i in range(idx):
    if i==0:
        tmp = np.load("./cam0_"+str(i)+".npy",allow_pickle=True)
    else:
        tmp = np.concatenate((tmp,np.load("./cam0_"+str(i)+".npy",allow_pickle=True)),axis=0)
    np.save("cam0_total",tmp)

for i in range(idx):
    if i==0:
        tmp = np.load("./cam1_"+str(i)+".npy",allow_pickle=True)
    else:
        tmp = np.concatenate((tmp,np.load("./cam1_"+str(i)+".npy",allow_pickle=True)),axis=0)
    np.save("cam1_total",tmp)

for i in range(idx):
    if i==0:
        tmp = np.load("./cam2_"+str(i)+".npy",allow_pickle=True)
    else:
        tmp = np.concatenate((tmp,np.load("./cam2_"+str(i)+".npy",allow_pickle=True)),axis=0)
    np.save("cam2_total",tmp)

for i in range(idx):
    if i==0:
        tmp = np.load("./cam3_"+str(i)+".npy",allow_pickle=True)
    else:
        tmp = np.concatenate((tmp,np.load("./cam3_"+str(i)+".npy",allow_pickle=True)),axis=0)
    np.save("cam3_total",tmp)
'''
cam_num = input("카메라 번호를 입력하세요 : ")
data = np.load("./cam"+cam_num+"_total.npy",allow_pickle=True)
#for i in range(int(len(data)/10)):
for i in range(3):
    plt.imshow(data[i*10])
    plt.show()
    plt.close('Figure 1')
    





