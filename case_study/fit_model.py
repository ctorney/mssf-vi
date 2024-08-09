# import modules
import os
import numpy as np
import random
from datetime import datetime

import math
import pandas as pd
import scipy.stats as stats
# import seaborn as sns
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
# import arviz as az
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
np.set_printoptions(suppress=True)
sys.path.append(".")
from mssf import ConvergenceCallback
from mssf import stepSelectionVI

import rioxarray as rxr
import rasterio
from rasterio.plot import plotting_extent
import xarray as xr
import geopandas as gpd
np.set_printoptions(suppress=True)
##
# Import data
FH=pd.read_csv("../data/amt_fisher.csv",low_memory=False)
covars = rxr.open_rasterio("../data/amt_covars_merged.tif",masked=True)

##

# Visualisation
fig,ax=plt.subplots(2,2,figsize=(12,10))
covars.sel(band=1).plot(robust=True,cmap="viridis",ax=ax[0,0],
       cbar_kwargs={"label":"Population density","pad":0.015,"shrink":1})

ax[0,0].plot(FH["X"],FH["Y"],".",color="black",alpha=0.5)
# ax[0,0].plot(sp[:,0], sp[:,1],".",color="black",alpha=0.5)

covars.sel(band=2).plot(robust=True,cmap="viridis",ax=ax[0,1],
       cbar_kwargs={"label":"Elevation","pad":0.015,"shrink":1})
ax[0,1].plot(FH["X"],FH["Y"],".",color="black",alpha=0.5)
# ax[0,1].plot(sp[:,0], sp[:,1],".",color="black",alpha=0.5)

covars.sel(band=3).plot(robust=True,cmap="viridis",ax=ax[1,0],
       cbar_kwargs={"label":"Grass","pad":0.015,"shrink":1})
ax[1,0].plot(FH["X"],FH["Y"],".",color="black",alpha=0.5)
# ax[1,0].plot(sp[:,0], sp[:,1],".",color="black",alpha=0.5)

covars.sel(band=4).plot(robust=True,cmap="viridis",ax=ax[1,1],
       cbar_kwargs={"label":"Wet","pad":0.015,"shrink":1})
ax[1,1].plot(FH["X"],FH["Y"],".",color="black",alpha=0.5)
# ax[1,1].plot(sp[:,0], sp[:,1],".",color="black",alpha=0.5)

ax[0,0].set_axis_off()
ax[0,1].set_axis_off()
ax[1,0].set_axis_off()
ax[1,1].set_axis_off()

ax[0,0].set_title("")
ax[0,1].set_title("")
ax[1,0].set_title("")
ax[1,1].set_title("")

plt.tight_layout()
plt.show()
##

covars[0] = covars[0] - np.mean(covars[0])
covars[0] = covars[0]/np.std(covars[0])

covars[1] = covars[1] - np.mean(covars[1])
covars[1] = covars[1]/np.std(covars[1])

##

# Convert date to recognized format
def make_Date(df):
    return datetime.strptime(df["Date"],"%Y-%m-%d %H:%M:%S")
FH["Timestamp"]=FH.apply(make_Date,axis=1)

# Filter only lupe as in the paper
FH=FH[FH["name"]=="Lupe"]
XY=FH[["X","Y"]].values
##

# covars_np = covars.values


##
# # Preprocessing raster layers by interpolating NAS
# Elev=Elev.rio.interpolate_na(method="nearest")
# Pop=Pop.rio.interpolate_na(method="nearest")


# ##
# coorddf = pd.DataFrame(XY, columns=['x', 'y'])
# coords = xr.Dataset.from_dataframe(coorddf)
# # find the elevation and population density at the points
# UsedElev=Elev.interp(coords)
# UsedPop=Pop.interp(coords)

# ##
# # Normalize the covariates
# Elev=Elev- np.mean(Elev)
# Pop=Pop - np.mean(Pop)
# Elev = Elev/np.std(Elev)
# Pop = Pop/np.std(Pop)

# # Stack the building together
# Cov=np.stack((Elev,Pop,Grass,Wet))
# Cov=np.stack((Grass))

# Cov=Cov.squeeze()

##

##


# preprocessing steps convert to tensors and handle the change of ID, start/end points and time between fixes


# Convert time to hours
secs =(pd.to_datetime(FH["Timestamp"])- datetime.strptime("1970-01-01 00:00:00","%Y-%m-%d %H:%M:%S")).dt.total_seconds().values

T = (secs/60)

 
ID = FH["id"].values

start_points = []
end_points = []
step_times = []

for i in np.unique(ID):
    cxy = XY[ID==i]
    cdt = T[ID==i]
    cdt=np.diff(cdt)
    if cdt.shape[0]<2:
        continue
    sp = cxy[:-1]
    ep = cxy[1:]
    dt = np.atleast_2d(cdt).T
    sp = sp[dt[:,0]<=2.33]
    ep = ep[dt[:,0]<=2.33]
    dt = dt[dt[:,0]<=2.33]
    sp = sp[dt[:,0]>=1.66]
    ep = ep[dt[:,0]>=1.66]
    dt = dt[dt[:,0]>=1.66]
    start_points.append(sp)
    end_points.append(ep)
    step_times.append(dt)

indexes = np.arange(np.vstack(start_points).shape[0])
np.random.shuffle(indexes)
start_points = tf.convert_to_tensor(np.vstack(start_points)[indexes],dtype=tf.float32)
end_points = tf.convert_to_tensor(np.vstack(end_points)[indexes],dtype=tf.float32)
step_times = tf.convert_to_tensor(np.vstack(step_times)[indexes],dtype=tf.float32)
cov_tensor = tf.transpose(tf.convert_to_tensor(covars,dtype=tf.float32),[2,1,0])
xlim = covars.rio.bounds()
x_ref_min= [xlim[0],xlim[1]]
x_ref_max= [xlim[2],xlim[3]]
##


diffs = end_points - start_points
diffs = diffs/np.sqrt(step_times)
move_std = np.std(diffs)

##

plt.hist(diffs[:,0], bins=100,density=True)
plt.plot(np.linspace(-150,150,100),stats.norm.pdf(np.linspace(-150,150,100),0,move_std))
plt.show()

##

# Create the instance of the model and set up the training
ssf = stepSelectionVI(4, cov_tensor, x_ref_min, x_ref_max, move_std=move_std, n_gh_points=3, n_gh_points_vi=5)

# set up the dataset and optimizer
batch_size = 1671
train_dataset = tf.data.Dataset.from_tensor_slices((start_points, end_points, step_times))
dataset_repeat = int(1e5/(batch_size))
train_dataset = train_dataset.shuffle(buffer_size=100000).batch(batch_size, drop_remainder=True).repeat(dataset_repeat)


overwrite_ema = start_points.shape[0]//batch_size

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, clipvalue=2.0, use_ema=True, ema_overwrite_frequency=overwrite_ema)
kl_weight = batch_size/start_points.shape[0]
ssf.compile(optimizer=optimizer, loss_weights=kl_weight)


convergence_callback = ConvergenceCallback(threshold=1e-3)

##
max_epochs = 500
ssf.fit(train_dataset, epochs=max_epochs, callbacks=[convergence_callback])


##

# save the results

np.save("beta_mean.npy",ssf.beta_mean.numpy())
np.save("beta_std.npy",ssf.beta_std.numpy())
np.save("move_std.npy",ssf.move_std.numpy())

##
# plot the normal posteriors

fig,ax=plt.subplots(2,2,figsize=(12,10))


for i in range(4):

    mean = ssf.beta_mean.numpy()[i]
    std = ssf.beta_std.numpy()[0,i]
    x = np.linspace(mean-3*std,mean+3*std,100)
    y = stats.norm.pdf(x,mean,std)
    ax[i//2,i%2].plot(x,y)
    ax[i//2,i%2].fill_between(x,y,0,alpha=0.5)
    ax[i//2,i%2].set_title(f"Posterior for beta_{i+1}")


plt.tight_layout()
plt.show()

