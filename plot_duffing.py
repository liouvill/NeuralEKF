# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:19:26 2022

@author: Liu Wei
"""

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


data_true=sio.loadmat("./duffing_data.mat")["true_latent"][...,:2]
data_noise=sio.loadmat("./duffing_data.mat")["Obs"]
y_s=np.load("./y_s.npy")
y_p=np.load("./y_p.npy")
latent_true=sio.loadmat("./duffing_data.mat")["true_latent"][...,:4]
z_s=np.load("./z_s.npy")
z_p=np.load("./z_p.npy")
y_NEKF=np.concatenate((y_s,y_p[1:]),axis=0)
z_NEKF=np.concatenate((z_s,z_p[1:]),axis=0)
print(latent_true.shape)
print(z_NEKF.shape)

fig, axs = plt.subplots(2,1,figsize=(70,50))
plt.rc('font', family='Times New Roman')
plt.rcParams["mathtext.fontset"] = "cm"
for i in range(2):
    #axs[0].set_title(f"$a_i$}")
    if i==0 or i==1:
        axs[i].set_ylabel(f"$x_{i%2+1}$", fontdict={'family' : 'Times New Roman', 'size' : 230})
        axs[i].plot(data_true[1000,:,i%2], color="black", linestyle="-", label="noise-free response", lw=20)
        axs[i].plot(data_noise[1000,:,i%2], color="gray", linestyle=(0, (1, 1)), label="noisy measurement", lw=20)
        axs[i].plot(y_NEKF[:,0,i%2], color="red", linestyle="--", label="Neural EKF", lw=20)
    #if i ==2 or i==3:
    #    axs[i//2,i%2].set_ylabel(f"$x_{i%2+1}$", fontdict={'family' : 'Times New Roman', 'size' : 120})
    #    axs[i//2,i%2].plot(data_true[0,:,i%2], color="black", linestyle="-", label="true response", lw=15)
    #    axs[i//2,i%2].plot(data_NEKF[:,0,i%2], color="red", linestyle="--", label="prediction", lw=15)
    #    axs[i//2,i%2].set_xlabel("Time [sec]", fontdict={'family' : 'Times New Roman', 'size' : 120})
    handles, labels = axs[i].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if i == 0: #or i==2:
        axs[i].legend(by_label.values(), by_label.keys(), loc="lower left", bbox_to_anchor= (0.0, 1.01), ncol= 2, prop={'family' : 'Times New Roman', 'size' : 190})
        axs[i].set_yticklabels([-1.0,-0.5,0,0.5,1.0], fontdict={'family' : 'Times New Roman', 'size' : 230})
        axs[i].set_yticks([-1.0,-0.5,0,0.5,1.0])
    if i == 1: #or i==3:
        axs[i].set_xlabel("Time [sec]", fontdict={'family' : 'Times New Roman', 'size' : 230})
        axs[i].set_yticklabels([-0.6,-0.4,-0.2,0,0.2,0.4], fontdict={'family' : 'Times New Roman', 'size' : 230})
        axs[i].set_yticks([-0.6,-0.4,-0.2,0,0.2,0.4])
           #axs[i].legend(loc="upper right", ncol= 1, prop={'family' : 'Times New Roman', 'size' : 60})
           #axs[i].tick_params(axis="x", labelsize=60)
    axs[i].set_xticks([0,10,20,30,40,50])
    axs[i].set_xticklabels([0,2,4,6,8,10], fontdict={'family' : 'Times New Roman', 'size' : 230})

#        temp = np.max(np.abs(data_true[1,:,i%2]))
#        axs[i//2,i%2].set_ylim((-1.05*temp, 1.05*temp))
plt.tight_layout()
plt.close()
fig.savefig("comparison")


fig, axs = plt.subplots(2,1,figsize=(50,75))
plt.rc('font', family='Times New Roman')
plt.rcParams["mathtext.fontset"] = "cm"
axs[0].plot(latent_true[1000,:,0], latent_true[1000,:,2], color="black", lw=15)
axs[1].plot(z_NEKF[:,0,0], z_NEKF[:,0,2], lw=15)
axs[0].set_xlabel(f"$x_1$", fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[0].set_ylabel(r"$\dot{x}_1$", fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[1].set_xlabel(f"$z_1$", fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[1].set_ylabel(f"$z_3$", fontdict={'family' : 'Times New Roman', 'size' : 200})
# plot seting for noise 0.001
# =============================================================================
# axs[0].set_xticklabels([-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.00], fontdict={'family' : 'Times New Roman', 'size' : 200})
# axs[0].set_xticks([-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.00])
# axs[0].set_yticklabels([-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0], fontdict={'family' : 'Times New Roman', 'size' : 200})
# axs[0].set_yticks([-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0])
# axs[1].set_xticklabels([-7.5,-5.0,-2.5,0,2.5,5.0,7.5], fontdict={'family' : 'Times New Roman', 'size' : 200})
# axs[1].set_xticks([-7.5,-5.0,-2.5,0,2.5,5.0,7.5])
# axs[1].set_yticklabels([-20,-15,-10,-5.0,0,5.0,10,15], fontdict={'family' : 'Times New Roman', 'size' : 200})
# axs[1].set_yticks([-20,-15,-10,-5.0,0,5.0,10,15])
# =============================================================================
# plot setting for noise 0.1
axs[0].set_xticklabels([-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.00], fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[0].set_xticks([-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.00])
axs[0].set_yticklabels([-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0], fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[0].set_yticks([-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0])
axs[1].set_xticklabels([-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.00], fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[1].set_xticks([-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.005])
axs[1].set_yticklabels([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8], fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[1].set_yticks([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8])
plt.tight_layout()
plt.close()
fig.savefig("phase")

fig, axs = plt.subplots(2,1,figsize=(50,75))
plt.rc('font', family='Times New Roman')
plt.rcParams["mathtext.fontset"] = "cm"
T,_,_,_= np.linalg.lstsq(z_NEKF[:,0,:], latent_true[1000,:,:])#, rcond=None)
z_NEKF_rot = z_NEKF @ T
axs[0].plot(latent_true[1000,:,0], latent_true[1000,:,2], color="black", lw=15)
axs[1].plot(z_NEKF_rot[:,0,0], z_NEKF_rot[:,0,2], lw=15)
axs[0].set_xlabel(f"$x_1$", fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[0].set_ylabel(r"$\dot{x}_1$", fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[1].set_xlabel(f"rotated $z_1$", fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[1].set_ylabel(f"rotated $z_3$", fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[0].set_xticklabels([-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.00], fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[0].set_xticks([-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.00])
axs[0].set_yticklabels([-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0], fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[0].set_yticks([-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0])
axs[1].set_xticklabels([-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.00], fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[1].set_xticks([-0.75,-0.50,-0.25,0,0.25,0.50,0.75,1.00])
axs[1].set_yticklabels([-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0], fontdict={'family' : 'Times New Roman', 'size' : 200})
axs[1].set_yticks([-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0])
plt.tight_layout()
plt.close()
fig.savefig("phase_rot")