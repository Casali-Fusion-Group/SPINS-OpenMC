# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:26:44 2025

@author: jakeb
"""

import ufiles
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy import interpolate
import matplotlib.cm as cm

def plot_data(u_file):
    rho=u_file["X"]["data"]
    times=u_file["Y"]["data"]
    data=u_file["f"]["data"]
    
    cmap = cm.plasma  # Choose a blue colormap
    norm = Normalize(vmin=np.min(times)-0.5, vmax=np.max(times)+0.5, clip=False)
    
    fix, axs = plt.subplots()
    
    
    for i,time in enumerate(times):
        color = cmap(norm(time))
        if i% 10 == 0:
            axs.plot(rho, data[:,i],color=color, label=f"t={time:.2f}s")
        else:
            axs.plot(rho, data[:,i],color=color)
    
    axs.legend()
    plt.show()
    
def plot_smooth_data(u_files, scales, measurements, colors, dt = 0.3):
    if len(u_files) != len(measurements) != len(colors):
        raise Exception("Number of u_files must equal number of measurements")
    
    fig, axs = plt.subplots(1,(len(u_files)),figsize = (8,4),dpi=300)
    
    fig.suptitle("Ion Density and Temperature Profiles")
    
    for i,(file, measurement) in enumerate(zip(u_files,measurements)):
        times=file["X"]["data"]
        rho=file["Y"]["data"]
        data=file["f"]["data"]*scales[i]
        
        profile_time = 0.500
        time_closest_index = np.argmin(np.abs(times - profile_time))
        time_range = 6
        
        # take closest time range of data
        data_t_trimmed = data[time_closest_index-int(time_range/2):time_closest_index+int(time_range/2),:]
        time_range = data_t_trimmed.shape[0]
        # temporally average profile 
        data_t_average = np.sum(data_t_trimmed,axis = 0)/time_range
        
        # trim off excess data beyond the LCFS (rho > 1.0)
        data_rho = np.stack((rho,data_t_average),axis=1) #np.array([rt for rt in Ti_profile if rt[0] <= 1.00])
        data_rho_trimmed = np.array([rn for rn in data_rho if rn[0] <= 1.00]) # trims off pedestal region
        data_rho_trimmed = data_rho_trimmed[data_rho_trimmed[:,0].argsort(),:]
        data_x = data_rho_trimmed[:,0]
        data_y = data_rho_trimmed[:,1]
        
        # create spline fit of temporally averaged profile
        data_spline = interpolate.UnivariateSpline(data_x, data_y, k = 1) #,s = Ti_y[0]**2.15 )
        
        # create all of the background traces to be plotted to show smoothing
        data_traces_rho = np.stack((np.vstack([data_rho[:,0]]*time_range),data_t_trimmed),axis=2) #np.array([rt for rt in Ti_profile if rt[0] <= 1.00])
        data_traces_rho_trimmed = np.zeros((data_t_trimmed.shape[0], data_rho_trimmed.shape[0], 2))
        for j,time_trace in enumerate(data_traces_rho):
            data_traces_rho_trimmed[j,:,:] = np.array([rn for rn in time_trace if rn[0] <= 1.00]) 
        
        xnew = np.linspace(0, max(data_x), num = 1000)
        
        cmap = cm.get_cmap(f"{colors[i]}")
        
        pad = 10
        norm = Normalize(vmin=-pad, vmax=time_range+pad)
        # Plot Ion Temperature Profile
        for j,time_trace in enumerate(data_traces_rho_trimmed):
            x = time_trace[:,0]
            y = time_trace[:,1]
            color = cmap(norm(j))
            if j == 0 or j == int(time_range - 1):
                axs[i].scatter(x, y, label=f't={times[int(time_closest_index-time_range/2 + j)]:1.2}s', color = color,s=7, alpha=0.25)
                #ax[0].plot(x, y, '-', label=f't={Ti_time[int(time_closest_index-time_range/2 + i)]:1.2}s', color = color)
            else:
                axs[i].scatter(x, y, color = color,s=7, alpha=0.25)
                #ax[0].plot(x, y, '-', color = color)
        
        smooth_label = r"$\overline{T}_{time \ avg}$" + f", t={times[int(time_closest_index)]:1.2}s"
        color=cmap(norm(time_range+4))
        axs[i].scatter(data_x, data_y, label=smooth_label, color = color, s = 15)
        axs[i].plot(xnew, data_spline(xnew), '-', c=color, label="Ti Interpolation")
        axs[i].legend()
        axs[i].set_xlabel(r"$\rho$")
        axs[i].set_ylabel(rf"{measurements[i]}")
        
    plt.tight_layout()
    plt.show()

uf2 = ufiles.UFILE(fin="Data/OMFITTER49392.txt")
uf3 = ufiles.UFILE(fin="Data/NER49392.txt")
uf4 = ufiles.UFILE(fin="Data/OMFITTI249392.txt")


# plot_data(uf2)
# plot_data(uf3)
# plot_data(uf4)

plot_smooth_data([uf4,uf3], [1000,1e6/1.3], [r"$T_i [eV]$",r"$n_e [m^{-3}]$"], ["Reds", "Blues"])

uf5 = ufiles.UFILE(fin="Data/OMF49392.NTX")

time=uf5["X"]["data"]
rate=uf5["f"]["data"]
plt.scatter(time,rate)
plt.show()

profile_time = 0.500
time_closest_index = np.argmin(np.abs(time - profile_time))
print(f"Neutron rate closest to the time slice is {rate[time_closest_index]:.2e}")


# Smooth data

kernal_size = 1000
kernal = np.ones(kernal_size) / kernal_size
rate_smoothed = np.convolve(rate, kernal, mode="same")

time_closest_index = np.argmin(np.abs(time - profile_time))
print(f"Neutron rate closest to the time slice after smoothing {rate_smoothed[time_closest_index]:.2e}")
fig, ax = plt.subplots(1,1,figsize = (8,4),dpi=1200)
fig.suptitle("Fission Chamber Total Neutron Rate")
ax.scatter(time, rate_smoothed,s=0.1)
ax.plot([0.5,0.5],[0.0,1.1e14],"k--")
ax.set_xlabel("Times (s)")
ax.set_ylabel("Neutron Count")
ax.set_xlim(0.1,1.0)
ax.set_ylim(0,1.1e14)


