# import modules
import os
import numpy as np
import random
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

from mssf_model import stepSelectionVI
from mssf_model import ConvergenceCallback

# Load data


nbObs = 10001  # nbObs Number of observations
beta = np.array([[0.5, -0.8]])  # Vector of resource selection coefficients
L = 50.0

pos_filename = "pos_N_" + str(nbObs-1) + "_b1_" + str(beta[0][0]) + "_b2_" + str(beta[0][1]) + "_repeat_0.csv"
grid_filename = "cov_N_" + str(nbObs-1) + "_b1_" + str(beta[0][0]) + "_b2_" + str(beta[0][1]) + "_repeat_0.npy"


# get home directory from os
home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "workspace", "mssf-vi")


df = pd.read_csv(data_dir + "/synthetic_data/" + pos_filename)

xy1 = df[["x", "y"]].values 
dt = df["time"].values  
ID = df["ID"].values  

cov_cube = np.load(data_dir + "/synthetic_data/" + grid_filename)
cov_cube = cov_cube[0]

##

# Process to convert to steps

# preprocessing steps convert to tensors and handle the change of ID, start/end points and time between fixes
start_points = []
end_points = []
step_times = []
for i in np.unique(ID):
    cxy = xy1[ID == i]  # [:1024*100+1]
    cdt = dt[ID == i]  # [:1024*100+1]
    if cdt.shape[0] < 2:
        continue
    start_points.append(cxy[:-1])
    end_points.append(cxy[1:])
    step_times.append(np.atleast_2d(cdt[:-1]).T)


indexes = np.arange(np.vstack(start_points).shape[0])
np.random.shuffle(indexes)
start_points = tf.convert_to_tensor(np.vstack(start_points)[indexes], dtype=tf.float32)
end_points = tf.convert_to_tensor(np.vstack(end_points)[indexes], dtype=tf.float32)
step_times = tf.convert_to_tensor(np.vstack(step_times)[indexes], dtype=tf.float32)

cov_tensor = tf.transpose(tf.convert_to_tensor(cov_cube, dtype=tf.float32), [2, 1, 0])

x_ref_min = [0., 0.]
x_ref_max = [L, L]

##

# Create the instance of the model and set up the training

ssf = stepSelectionVI(2, cov_tensor, move_std=2.00,L=50.0, n_gh_points=3, n_gh_points_vi=5)

# set up the dataset and optimizer
batch_size = 1000
train_dataset = tf.data.Dataset.from_tensor_slices((start_points, end_points, step_times))
train_dataset = train_dataset.batch(batch_size,drop_remainder=True)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, use_ema=True, ema_overwrite_frequency=10) 

kl_weight = batch_size/start_points.shape[0]
ssf.compile(optimizer=optimizer, loss_weights=kl_weight)

convergence_callback = ConvergenceCallback(threshold=1e-3)



##
max_epochs = 500
ssf.fit(train_dataset, epochs=max_epochs, callbacks=[convergence_callback])
##
