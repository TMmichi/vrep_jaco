import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt
from matplotlib import animation

data_ori = np.load("./dummy_data.npy",allow_pickle=True)
data_vae = np.load("./vae_result_1e5_500.npy",allow_pickle=True)
fig = plt.figure(frameon=False)
plt.axis('off')

ims = []

for i in range(int(len(data_vae)/3)):
    plt.subplot(211)
    plt.imshow(data_ori[0][0][0])
    plt.subplot(212)
    im = plt.imshow(data_vae[i])
    #plt.show()
    ims.append([im])


ani = animation.ArtistAnimation(fig, ims, interval=500)
ani.save('test.mp4',dpi=300)

