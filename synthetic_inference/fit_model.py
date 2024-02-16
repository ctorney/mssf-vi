# import modules
from mssf_model import stepSelectionVI
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


# Load data


nbObs = 10001  # nbObs Number of observations
beta = np.array([[0.5, -0.8]])  # Vector of resource selection coefficients
L = 50.0

pos_filename = "pos_N_" + \
    str(nbObs-1) + "_b1_" + str(beta[0][0]) + \
    "_b2_" + str(beta[0][1]) + "_repeat_0.csv"
grid_filename = "cov_N_" + \
    str(nbObs-1) + "_b1_" + str(beta[0][0]) + \
    "_b2_" + str(beta[0][1]) + "_repeat_0.npy"


# get home directory from os
home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "workspace", "mssf-vi")


df = pd.read_csv(data_dir + "/synthetic_data/" + pos_filename)

xy1 = df[["x", "y"]].values  # ,["y"]].values
dt = df["time"].values  # ,["y"]].values
ID = df["ID"].values  # ,["y"]].values
# df

# print(xy1.shape)
# print(cov_cube.shape)
cov_cube = np.load(data_dir + "/synthetic_data/" + grid_filename)
cov_cube = cov_cube[0]
print(cov_cube.shape)

##

# ## Process to convert to steps

# In[3]:


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
start_points = tf.convert_to_tensor(
    np.vstack(start_points)[indexes], dtype=tf.float32)
end_points = tf.convert_to_tensor(
    np.vstack(end_points)[indexes], dtype=tf.float32)
step_times = tf.convert_to_tensor(
    np.vstack(step_times)[indexes], dtype=tf.float32)


cov_tensor = tf.transpose(tf.convert_to_tensor(
    cov_cube, dtype=tf.float32), [2, 1, 0])

x_ref_min = [0., 0.]
x_ref_max = [L, L]

##

# ## Create the instance of the model and set up the training

# In[4]:


ssf = stepSelectionVI(2, cov_tensor, move_std=2.00,
                      L=50.0, n_gh_points=3, n_gh_points_vi=5)


# set up the dataset and optimizer
batch_size = 1000
train_dataset = tf.data.Dataset.from_tensor_slices(
    (start_points, end_points, step_times))
# train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
# train_dataset = train_dataset.shuffle(
#     buffer_size=100000).batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.batch(batch_size,drop_remainder=True)

kl_weight = batch_size/start_points.shape[0]
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.1, decay_steps=10, decay_rate=0.9, staircase=False)
##

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, use_ema=True, ema_overwrite_frequency=10, ema_momentum=0.99)
                                     #, amsgrad=True, beta_1 = 0.99)

# optimizer = tfp.optimizer.VariationalSGD(batch_size, total_num_examples=start_points.shape[0], max_learning_rate=1.0, burnin=20, burnin_max_learning_rate=0.1)
# clip value is wrong because it's bouncing back and forth
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01,clipvalue=10.0)  # , beta_1 = 0.5)
ssf.compile(optimizer=optimizer, loss_weights=kl_weight)

# add the diff norm metric
#https://github.com/pymc-devs/pymc/blob/0d8ddbace0056bdeae01d706e52ac11baf2cf3f3/pymc/variational/callbacks.py#L44 
##
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=100) #, min_delta=0.001)
convergence_callback = ConvergenceCallback(1e-3)
ssf.fit(train_dataset, epochs=500, callbacks=[convergence_callback])
##
# In[6]:

chain1 = np.load('chain1.npy')


def plot_fit1():  # data_batch
    print('move std ' + str(ssf.move_std.numpy()) + 'betas ' +
          str(ssf.beta_mean.numpy()) + ' stds: ', str(ssf.beta_std.numpy().flatten()))

    b1mean = ssf.beta_mean.numpy()[0]
    b1scale = ssf.beta_std.numpy()[0, 0]
    b2mean = ssf.beta_mean.numpy()[1]
    b2scale = ssf.beta_std.numpy()[0, 1]

    x = np.arange(b1mean-b1scale*5, b1mean+b1scale*5, 0.001)
    y = tfp.distributions.Normal(loc=b1mean, scale=b1scale).prob(x)

    plt.plot(x, y, c='C0')
    plt.hist(chain1[:, 0, 0], bins=10, density=True, color='C1')
    plt.fill_between(x, y, color='C0', alpha=0.5)
    plt.axvline(0.5, c='C1')
    # ax[0].set_ylim(0,y.numpy().max()*1.2)
    plt.show()
    return
def plot_fit2():  # data_batch
    print('move std ' + str(ssf.move_std.numpy()) + 'betas ' +
          str(ssf.beta_mean.numpy()) + ' stds: ', str(ssf.beta_std.numpy().flatten()))

    b1mean = ssf.beta_mean.numpy()[0]
    b1scale = ssf.beta_std.numpy()[0, 0]
    b2mean = ssf.beta_mean.numpy()[1]
    b2scale = ssf.beta_std.numpy()[0, 1]

    x = np.arange(b2mean-b2scale*5, b2mean+b2scale*5, 0.001)
    y = tfp.distributions.Normal(loc=b2mean, scale=b2scale).prob(x)

    plt.plot(x, y, c='C0')
    plt.hist(chain1[:, 0, 1], bins=20, density=True, color='C1')
    plt.fill_between(x, y, color='C0', alpha=0.5)

    plt.axvline(-0.8, c='C1')
    # plt.ylim(0,y.numpy().max()*1.2)
    # plt.savefig('1millionpoints.png')
    plt.show()


##
plot_fit()
##
# add early stopping

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=3, min_delta=0.1)

ssf.fit(train_dataset, epochs=50)  # , callbacks=[early_stopping])
# calculate the correlation between chain[:,0,0] and chain[:,0,1]
# cor = np.corrcoef(chain1[:,0,0],chain1[:,0,1])[0,1]
##

epochs = range(5)

for epoch in epochs:
    epoch_loss = 0.0
    for data_batch in tqdm(train_dataset):
        with tf.GradientTape() as tape:
            loss = ssf.variational_loss(*data_batch, kl_weight)
        epoch_loss += np.squeeze(loss.numpy())
        gradients = tape.gradient(loss, ssf.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ssf.trainable_variables))
        # break
    # plot_fit()
    # if epoch % 1 == 0:
        # print('Epoch ' + str(epoch) + ' complete. Loss: ', epoch_loss)
        # print('Epoch ' + str(epoch) +' betas ' + str(ssf2.beta_mean.numpy()) + ' stds: ', str(ssf2.beta_std.numpy().flatten()))
