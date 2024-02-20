from mssf import stepSelectionVI
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


##
output_filename = "model_hmc_results.csv"

# write the header
with open(output_filename, "w") as f:
    f.write("nbObs, truebeta1, truebeta2, betamean1, betamean2, betastd1, betastd2\n")


def fit_dataset(nbObs, beta):

    ##
    # Load data
    L = 50.0
    dtype="float32"
    Ncov=2 # Number of covariates
    repeat = 0

    pos_filename = "pos_N_" + str(nbObs-1) + "_b1_" + str(beta[0][0]) + "_b2_" + str(beta[0][1]) + "_repeat_" + str(repeat) + ".csv"
    grid_filename = "cov_N_" + str(nbObs-1) + "_b1_" + str(beta[0][0]) + "_b2_" + str(beta[0][1]) + "_repeat_" + str(repeat) + ".npy"

    # get home directory from os
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, "workspace", "mssf-vi")

    df = pd.read_csv(data_dir + "/data/" +  pos_filename)

    xy1 = df[["x", "y"]].values
    dt = df["time"].values
    ID = df["ID"].values

    cov_cube = np.load(data_dir + "/data/" + grid_filename)


    # preprocessing steps convert to tensors and handle the change of ID, start/end points and time between fixes
    start_points = []
    end_points = []
    step_times = []
    for i in np.unique(ID):
        cxy = xy1[ID == i] 
        cdt = dt[ID == i] 
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

    # Create the instance of the model and set up the training
    ssf = stepSelectionVI(2, cov_tensor, move_std=1.00,
                          L=50.0, n_gh_points=3, n_gh_points_vi=5)
    
    # Use the joint distribution to define the target log probability
    def target_log_prob(beta_params):
        return ssf.log_likelhood(beta_params,start_points,end_points,step_times) +tf.reduce_sum(tfd.Normal(loc=0.0,scale=10.0).log_prob(beta_params))

    b_start = tf.Variable(initial_value=[[0.0, 0.0]], dtype=tf.float32)
    
    num_results=int(1e4) 
    num_burnin_steps=int(1e3)
    
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=target_log_prob,
                                                     num_leapfrog_steps=3,step_size=0.1),
                                                     num_adaptation_steps=int(num_burnin_steps * 0.8))

    # add a progress bar
    adaptive_hmc = tfp.experimental.mcmc.WithReductions(
        adaptive_hmc,
        reducer=tfp.experimental.mcmc.ProgressBarReducer(
            num_results=num_results + num_burnin_steps,
        )
    )

##
    # Run the chain (with burn-in).
    @tf.function
    def run_chain():
        # Run the chain (with burn-in).
        samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=b_start,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)
        return samples

    
    # Run it and get samples from the posterior distribution for different chains.
    # Set number of chains
    n_chain=4
    samples_return=np.array([run_chain() for i in range(n_chain)])

    filename = "hmc_samples_" + str(nbObs) + "_b1_" + str(beta[0][0]) + "_b2_" + str(beta[0][1]) + ".npy"
    np.save(filename, samples_return)

    # Write the results to a file
    # nbObs, beta, repeat, betamean1, betamean2, betastd1, betastd2
    hmc_sample=tf.squeeze(samples_return)
    hmc_sample=tf.reshape(hmc_sample,(-1,2))
    b1mean=tf.math.reduce_mean(hmc_sample,axis=0).numpy()[0]
    b1scale=tf.math.reduce_std(hmc_sample,axis=0).numpy()[0]
    b2mean=tf.math.reduce_mean(hmc_sample,axis=0).numpy()[1]
    b2scale=tf.math.reduce_std(hmc_sample,axis=0).numpy()[1]

    with open(output_filename, "a") as f:
        f.write(str(nbObs) + "," + str(beta[0][0]) + "," + str(beta[0][1]) + "," + "," +
                str(b1mean) + "," + str(b2mean) + "," + str(b1scale) + "," + str(b2scale) + "\n")

beta_list = [[[0.5, -0.8]], [[-1.5, -1.8]], [[-1.5, 1.8]], [[1.2, 1.8]]]

for beta in beta_list:
    fit_dataset(10001, beta)
    print("Finished: ", beta)
