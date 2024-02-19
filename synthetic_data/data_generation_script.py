# import modules
import numpy as np
import random
import math
import pandas as pd

import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

np.set_printoptions(suppress=True)

# get home directory from os
home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "workspace", "mssf-vi", "data")

##

L = 50.0  # size of domain
x = y = np.arange(0, L+1, 1)
var = 1.0

tf.random.set_seed(time())

nrepeat = 10

# Make the environment using a correlated periodic Gaussian random field
x = y = np.arange(0, L+1, 1)
xx, yy = np.meshgrid(x, y)

positions = np.vstack([xx.ravel(), yy.ravel()]).T

len_scale_1 = 5
period = np.float64(L)
periodic_len = (2**0.5)*len_scale_1*np.pi/period


kernel = tfp.math.psd_kernels.ExpSinSquared(
    length_scale=periodic_len, period=period)
gp1 = tfd.GaussianProcess(kernel, positions)

len_scale_2 = 10
period = np.float64(L)
periodic_len = (2**0.5)*len_scale_2*np.pi/period


kernel = tfp.math.psd_kernels.ExpSinSquared(
    length_scale=periodic_len, period=period)
gp2 = tfd.GaussianProcess(kernel, positions)


@tf.function
def onestep(xyt, stdt, cov_field, beta_values):
    C = tf.random.normal(shape=(nrepeat, 2), mean=xyt, stddev=stdt[:, None])
    grid = tf.random.normal(shape=(nrepeat, npts, 2), mean=C[:, None, :], stddev=stdt[:, None, None])

    zgrid = grid % L
    cov_xy_end = tfp.math.batch_interp_regular_nd_grid(zgrid, x_min, x_max, cov_field, axis=-3)
    allrsf = tf.reduce_sum(tf.multiply(cov_xy_end, beta_values), axis=-1)

    inds = tfd.Categorical(logits=allrsf).sample()
    locations = tf.gather(grid, inds, batch_dims=1)
    return locations


for nbObs in [10001, 100001, 1000001]:
    for beta in [[[-1.5, -1.8]], [[0.5, -0.8]], [[-1.5, 1.8]], [[1.2, 1.8]]]:

        samples = gp1.sample(nrepeat)
        cov1 = samples.numpy().reshape(nrepeat, 51, 51)

        samples = gp2.sample(nrepeat)
        cov2 = samples.numpy().reshape(nrepeat, 51, 51)

        cov_cube = np.stack((cov1, cov2), axis=1)

        xy0 = [25, 25]  # Initial location
        npts = 100  # Number of potential endpoints to sample at each time step
        xy1 = np.zeros((nbObs, nrepeat, 2))
        xy1[0] = xy0
        x_min = [0., 0.]
        x_max = [L, L]

        # Convert the covariates to tensor
        cov_tensor = tf.transpose(tf.convert_to_tensor(
            cov_cube, dtype=tf.float32), [0, 3, 2, 1])
        beta_tensor = tf.convert_to_tensor(beta, dtype=tf.float32)

        dt = np.random.normal(loc=1.0, scale=0.1, size=(nrepeat, nbObs-1))
        ID = np.zeros(nbObs)

        dt = tf.convert_to_tensor(dt, dtype=tf.float32)
        stds = (0.5*dt)**0.5
        xy = tf.convert_to_tensor(xy1, dtype=tf.float32)

        for t in tqdm(range(1, nbObs)):

            next_points = onestep(xy[t-1], stds[:, t-1], cov_tensor, beta_tensor)
            tindex = tf.repeat(tf.constant(t, dtype=tf.int64), (20))
            xindex = tf.where(tf.ones_like(next_points))

            xy = tf.tensor_scatter_nd_update(xy, tf.concat(
                [tindex[:, None], xindex], axis=1), tf.reshape(next_points, (-1)))

        for repeat in range(nrepeat):
            pos_filename = data_dir + "/pos_N_" + str(nbObs-1) + "_b1_" + str(
                beta[0][0]) + "_b2_" + str(beta[0][1]) + "_repeat_" + str(repeat) + ".csv"
            grid_filename = data_dir + "/cov_N_" + str(nbObs-1) + "_b1_" + str(
                beta[0][0]) + "_b2_" + str(beta[0][1]) + "_repeat_" + str(repeat) + ".npy"

            # Saving the data to csv
            hj = pd.DataFrame(xy[:, repeat].numpy())
            hj.columns = ["x", "y"]
            date = pd.DataFrame(dt[repeat].numpy())
            date.columns = ["time"]
            Animal_ID = pd.DataFrame(ID)
            Animal_ID.columns = ["ID"]
            # Creating a list of data and merging them by columns
            simdata = [hj, date, Animal_ID]
            simdata = pd.concat(simdata, axis=1)
            # Saving the data
            simdata.to_csv(pos_filename, index=None)
            np.save(grid_filename, cov_cube[repeat])
