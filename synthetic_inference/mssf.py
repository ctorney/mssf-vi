import tensorflow as tf

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import numpy as np

# A periodic version for working with simulated data on a torus


class stepSelectionVI(tf.Module):
    def __init__(self, n_covars, cov_tensor, move_std=1.0, L=1.0,  prior_mean=None, prior_std=None,n_gh_points=4,n_vi_samples=4):
               
      
        self.n_covars = n_covars
        self.cov_tensor = cov_tensor
        self.n_gh_points = n_gh_points
        self.n_vi_samples = n_vi_samples
        
        self.x_ref_min = [0., 0.]
        self.x_ref_max = [L, L]
        self.L = L # lengths of square periodic domain

        # set up the points for Gauss-Hermite quadrature
        xi, wi = np.polynomial.hermite.hermgauss(n_gh_points)

        ghx,ghy = np.meshgrid(xi,xi)
        self.gh_grid = tf.convert_to_tensor(np.stack([ghx,ghy],axis=2).astype(np.float32)) 

        ghx,ghy = np.meshgrid(wi,wi)
        
        self.grid_gh_weights = tf.convert_to_tensor(ghy,dtype=tf.float32)
        self.gh_weights = tf.convert_to_tensor(wi,dtype=tf.float32)

        
        self.beta_mean = tf.Variable(np.zeros((n_covars)),dtype=tf.float32)
        
        #self.beta_mean = tf.Variable(np.array([-2.24205348,  4.03831697]),dtype=tf.float64)
        
        #self.beta_mean = tf.Variable(np.array([-1.5,1.8]),dtype=tf.float64)

        # self.beta_std = tfp.util.TransformedVariable([np.ones(n_covars,dtype=np.float32)],tfp.bijectors.Softplus(),dtype=tf.float32)
        self.beta_std = tfp.util.TransformedVariable([np.eye(n_covars,dtype=np.float32)],tfb.FillScaleTriL(),dtype=tf.float32)
        self.variational_posterior = tfp.distributions.MultivariateNormalTriL(loc = self.beta_mean, scale_tril=self.beta_std)
        # self.variational_posterior = tfp.distributions.MultivariateNormalDiag(loc = self.beta_mean, scale_diag=self.beta_std)
        
        self.move_std = tfp.util.TransformedVariable(move_std,tfp.bijectors.Softplus(),dtype=tf.float32, trainable=False)
        #self.variational_posterior = tfp.distributions.MultivariateNormalDiag(loc = self.beta_mean, scale_diag=self.beta_std)

        
        
        # set default prior to be N(0,10)
        if prior_mean is None:
            prior_mean = np.zeros((n_covars))
        if prior_std is None:
            prior_std = 10.0*np.ones((n_covars))
        prior_mean = tf.constant(prior_mean,dtype=tf.float32)
        prior_std = tf.constant(prior_std,dtype=tf.float32)
        self.prior = tfp.distributions.MultivariateNormalDiag(loc=prior_mean, scale_diag=prior_std)

    @tf.function
    def variational_loss(self, start_points_batch,end_points_batch,step_times_batch,kl_weight = 1.0):
        # beta_samples = tf.random.normal((self.n_vi_samples,self.n_covars,1),dtype=tf.float32)
        # beta_samples = (tf.expand_dims(self.beta_mean,axis=-1) + tf.linalg.matmul(self.beta_std,beta_samples))[...,0]
        beta_samples = self.variational_posterior.sample(self.n_vi_samples)[:,0,:]
        
        #beta_samples = tf.random.normal((self.n_vi_samples,self.n_covars),dtype=tf.float32)
        #beta_samples = self.beta_mean + self.beta_std*beta_samples
        elogp = tf.reduce_mean(self.log_likelhood(beta_samples,start_points_batch,end_points_batch,step_times_batch))
        penalty = kl_weight * tfp.distributions.kl_divergence(self.variational_posterior,self.prior)

        return -elogp+penalty
    
    
    # fast vectorised tf code
    @tf.function
    def log_likelhood(self, beta_params,start_points,end_points,step_times):
        # calculate the covariates at the step end locations using tensorflow probability
        env_points= tf.math.mod(end_points,self.L)

        cov_xy_end = tfp.math.batch_interp_regular_nd_grid(env_points, self.x_ref_min, self.x_ref_max, self.cov_tensor, axis=-3)
        # calculate the rsf
        rsf_points_end = tf.exp(tf.reduce_sum(tf.multiply(tf.expand_dims(cov_xy_end,0),tf.expand_dims(beta_params,1)),axis=-1))
        
        #sigma_var = 1.0 ## just for now - needs to be optimised too
        sigmas = tf.math.sqrt(self.move_std *0.5*step_times)

        def fn(input_locations):
            
            ## shapes needs to broadcast with (num_steps,ghx points, ghy points, 2)
            half_sigma = tf.reshape(sigmas,(-1,1,1,1))
            means = tf.reshape(input_locations,(-1,1,1,2))

            # shape is numvisamples, num steps, ghx,ghy, 1xnc matrix of coefficients
            betas = tf.reshape(beta_params,(-1,1,1,1,1,self.n_covars))
            gh_points = tf.math.mod(means+(2**0.5)*half_sigma*self.gh_grid,self.L)
            cov_gh_points  = tfp.math.batch_interp_regular_nd_grid(gh_points, self.x_ref_min, self.x_ref_max, self.cov_tensor, axis=-3)
            # need to add a dimension at the start to broadcast with num samples
            cov_gh_points = tf.expand_dims(tf.expand_dims(cov_gh_points,-1),0) 

            rsf =tf.exp(tf.matmul(betas,cov_gh_points)[...,0,0])

            # rsf is now dimension (numvisamples, numsteps, ghx, ghy)
            # multiply by the weights and sum to approximate the inner integral
            ysum = tf.reduce_sum(rsf*self.grid_gh_weights/(np.pi**0.5),axis=2)
            # multiply by the weights and sum to approximate the outer integral
            xsum = tf.reduce_sum(ysum*self.gh_weights/(np.pi**0.5),axis=2)
            return xsum

        half_sigma = tf.reshape(sigmas,(1,1,-1,1))


        # reshape to be ghx, ghx, num steps, 2
        mean_c = tf.reshape(0.5*(start_points+end_points),(1,1,-1,2))


        mid_points = mean_c + half_sigma*tf.expand_dims(self.gh_grid,2)

        inv_int_z = tf.math.pow(tf.map_fn(fn,tf.reshape(mid_points,(self.n_gh_points*self.n_gh_points,-1,2)),parallel_iterations=1),-1)


        inv_int_z = tf.reshape(inv_int_z,(self.n_gh_points,self.n_gh_points,tf.shape(inv_int_z)[1],tf.shape(inv_int_z)[2]))

        # inv_int_z is now shape (ghx,ghy,num vi samples, num steps)
        ysum = tf.reduce_sum(inv_int_z*tf.reshape(self.grid_gh_weights,(self.n_gh_points,self.n_gh_points,1,1))/(np.pi**0.5),axis=0)
        xsum = tf.reduce_sum(ysum*tf.reshape(self.gh_weights,(self.n_gh_points,1,1))/(np.pi**0.5),axis=0)


        step_log_prob = tfp.distributions.Independent(tfp.distributions.Normal(loc=start_points,scale=(2**0.5)*sigmas),reinterpreted_batch_ndims=1).log_prob(end_points)

        log_prob = tf.math.log(rsf_points_end) + tf.math.log(xsum) + step_log_prob
        return tf.math.reduce_sum(log_prob,axis=-1)
