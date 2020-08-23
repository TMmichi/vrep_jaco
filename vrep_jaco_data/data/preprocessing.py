import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
classifier = input("전처리라면 a , visualize라면 아무거나를 눌러주세요 : ")
if classifier=='a':
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
            os.remove("./cam0_"+str(i)+".npy")
        else:
            tmp = np.concatenate((tmp,np.load("./cam0_"+str(i)+".npy",allow_pickle=True)),axis=0)
            os.remove("./cam0_"+str(i)+".npy")
        np.save("./data/cam0_total",tmp)

    for i in range(idx):
        if i==0:
            tmp = np.load("./cam1_"+str(i)+".npy",allow_pickle=True)
            os.remove("./cam1_"+str(i)+".npy")
        else:
            tmp = np.concatenate((tmp,np.load("./cam1_"+str(i)+".npy",allow_pickle=True)),axis=0)
            os.remove("./cam1_"+str(i)+".npy")
        np.save("./data/cam1_total",tmp)

    for i in range(idx):
        if i==0:
            tmp = np.load("./cam2_"+str(i)+".npy",allow_pickle=True)
            os.remove("./cam2_"+str(i)+".npy")
        else:
            tmp = np.concatenate((tmp,np.load("./cam2_"+str(i)+".npy",allow_pickle=True)),axis=0)
            os.remove("./cam2_"+str(i)+".npy")
        np.save("./data/cam2_total",tmp)

    for i in range(idx):
        if i==0:
            tmp = np.load("./cam3_"+str(i)+".npy",allow_pickle=True)
            os.remove("./cam3_"+str(i)+".npy")            
        else:
            tmp = np.concatenate((tmp,np.load("./cam3_"+str(i)+".npy",allow_pickle=True)),axis=0)
            os.remove("./cam3_"+str(i)+".npy")
        np.save("./data/cam3_total",tmp)


else:
    cam_num = input("카메라 번호를 입력하세요 : ")
    data = np.load("./data/cam"+cam_num+"_total.npy",allow_pickle=True)
    a = input("사진 순번을 입력해주세요 : ")
    plt.imshow(data[int(a)],cmap='gray')
    plt.show()
    plt.close('Figure 1')
    





