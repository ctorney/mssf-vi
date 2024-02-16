import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class ConvergenceCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold,eps=1e-6):
        super().__init__()
        self.previous_variables = None
        self.eps = eps
        self.threshold = threshold


    def on_epoch_end(self, epoch, logs=None):
        if self.previous_variables is None:
            self.previous_variables = np.concatenate([v.numpy().flatten() for v in self.model.trainable_variables])
            return

        current_variables = np.concatenate([v.numpy().flatten() for v in self.model.trainable_variables])

        diff = current_variables - self.previous_variables  # type: ignore
        relative_diff = (np.abs(diff)) / (np.abs(self.previous_variables) + self.eps)

        self.previous_variables = current_variables

        norm = np.linalg.norm(relative_diff, np.inf)
        print("\nConvergence diagnostic: ", norm, "\n")
        if norm < self.threshold:
            self.stopped_epoch = epoch
            self.model.stop_training = True




class BinaryTruePositives(tf.keras.metrics.Metric):

      def __init__(self, name='binary_true_positives', tv=None, **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.gradients = [self.add_weight(name=v.name,shape=v.shape, initializer='ones') for v in tv]
        self.decay = 1. - 1. / tf.cast(10, np.float32)

      def update_state(self, gradients):

        grad_inner_product = sum(tf.reduce_sum(g1 * g2)/(tf.norm(g1)*tf.norm(g2))
                                  for g1, g2 in zip(self.gradients, gradients))


        # grad_inner_product = tf.reduce_sum(self.gradients[0][0] * gradients[0][0])
        # tf.print('\nvalues that go into it\n', self.gradients[0][0], gradients[0][0], output_stream=sys.stdout)
        self.true_positives.assign(self.decay*self.true_positives + (1-self.decay)*grad_inner_product)
        # self.true_positives.assign(grad_inner_product)
        # tf.print('\n', self.true_positives, grad_inner_product, output_stream=sys.stdout)
        for g1, g2 in zip(self.gradients, gradients):
          # tf.print('\n',g1, g2, output_stream=sys.stdout)
          g1.assign(g2)
        # # self.moving_product = (
        #     # grad_inner_product + self.decay * (self.moving_product - grad_inner_product))

        # self.previous_grads = gradients
        # y_true = tf.cast(y_true, tf.bool)
        # y_pred = tf.cast(y_pred, tf.bool)
        #
        # values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        # values = tf.cast(values, self.dtype)
        # if sample_weight is not None:
        #   sample_weight = tf.cast(sample_weight, self.dtype)
        #   sample_weight = tf.broadcast_to(sample_weight, values.shape)
        #   values = tf.multiply(values, sample_weight)
        # self.true_positives.assign_add(tf.reduce_sum(values))

      def result(self):
        return self.true_positives

# A periodic version for working with simulated data on a torus


class stepSelectionVI(tf.keras.Model):
    def __init__(self, n_covars, cov_tensor, move_std=1.0, L=1.0,  prior_mean=None, prior_std=None, n_gh_points=4, n_gh_points_vi=4, window_size=10):

        super().__init__()

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.n_covars = n_covars
        self.cov_tensor = cov_tensor
        self.n_gh_points = n_gh_points
        self.n_gh_points_vi = n_gh_points_vi

        self.x_ref_min = [0., 0.]
        self.x_ref_max = [L, L]
        self.L = L  # lengths of square periodic domain

        # set up the points for Gauss-Hermite quadrature
        xi, wi = np.polynomial.hermite.hermgauss(n_gh_points)

        ghx, ghy = np.meshgrid(xi, xi)
        self.gh_grid = tf.convert_to_tensor(
            np.stack([ghx, ghy], axis=2).astype(np.float32)) 

        ghx, ghy = np.meshgrid(wi, wi)

        self.grid_gh_weights = tf.convert_to_tensor(ghy, dtype=tf.float32) 
        self.gh_weights = tf.convert_to_tensor(wi, dtype=tf.float32) 

        # we have n_gh_points_vi in each dimension so the total number of points is n_gh_points_vi**n_covars
        self.n_vi_points = n_gh_points_vi**n_covars
        # set up the points for Gauss-Hermite quadrature for the variational inference
        xi, wi = np.polynomial.hermite.hermgauss(n_gh_points_vi)

        vi_grid = np.meshgrid(*[xi]*n_covars)
        self.gh_grid_vi = tf.convert_to_tensor(
            np.stack(vi_grid, axis=-1).astype(np.float32))

        self.gh_grid_vi = tf.reshape(self.gh_grid_vi, (-1, n_covars)) * (2**0.5)
        # ghx, ghy = np.meshgrid(wi, wi)
        #
        # n_gh_points_vi = 4
        # n_covars = 3
        # index_grid = np.meshgrid(*[np.arange(n_gh_points_vi)]*n_covars)

        # index_grid = np.stack(index_grid, axis=-1).astype(np.float32)

        weight_grid = np.meshgrid(*[wi]*n_covars)
        weight_grid = np.stack(weight_grid, axis=-1).astype(np.float32) / (np.pi**0.5)
        gh_grid_weights_vi = tf.convert_to_tensor(weight_grid, dtype=tf.float32)
        gh_grid_weights_vi = tf.math.reduce_prod(gh_grid_weights_vi, axis=-1) 

        self.gh_grid_weights_vi = tf.reshape(gh_grid_weights_vi, (-1)) 

        #


        self.beta_mean = tf.Variable(np.zeros((n_covars)), dtype=tf.float32)

        self.beta_std = tfp.util.TransformedVariable(
            [np.ones(n_covars, dtype=np.float32)/10.0], tfp.bijectors.Softplus(), dtype=tf.float32)

        self.variational_posterior = tfp.distributions.MultivariateNormalDiag(
            loc=self.beta_mean, scale_diag=self.beta_std)

        self.move_std = tfp.util.TransformedVariable(
            move_std, tfp.bijectors.Softplus(), dtype=tf.float32, trainable=True)

        # set default prior to be N(0,10)
        if prior_mean is None:
            prior_mean = np.zeros((n_covars))
        if prior_std is None:
            prior_std = 10.0*np.ones((n_covars))
        prior_mean = tf.constant(prior_mean, dtype=tf.float32)
        prior_std = tf.constant(prior_std, dtype=tf.float32)
        self.prior = tfp.distributions.MultivariateNormalDiag(
            loc=prior_mean, scale_diag=prior_std)

        self.window_size = window_size

        self.previous_grads = None
        self.converged = False

        # self.previous_grads = [tf.Variable(tf.ones_like(v), trainable=False)
                               # for v in self.trainable_variables]
        self.decay = 1. - 1. / tf.cast(self.window_size, np.float32)
        self.convergence_tracker = BinaryTruePositives(name="conv", tv=self.trainable_variables)

    def call(self):
        return self.variational_posterior

    def train_step(self, data):

        x, y, z = data
        kl_weight = self.compiled_loss._user_loss_weights

        with tf.GradientTape() as tape:
            loss = self.variational_loss(x, y, z, kl_weight)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.convergence_tracker.update_state(gradients)

        # grad_inner_product = sum(tf.reduce_sum(g1 * g2)
                                 # for g1, g2 in zip(self.previous_grads, gradients))
        # # self.moving_product = (
        #     # grad_inner_product + self.decay * (self.moving_product - grad_inner_product))

        # self.previous_grads = gradients

        return {"loss": self.loss_tracker.result(), "converged": self.convergence_tracker.result(), "beta std": self.beta_std[0][0], "beta mean": self.beta_mean[0]}

    @tf.function
    def variational_loss(self, start_points_batch, end_points_batch, step_times_batch, kl_weight=1.0):

        # beta_samples = self.variational_posterior.sample(self.n_vi_samples)[
            # :, 0, :]

        beta_values = self.beta_mean[None] + self.beta_std * self.gh_grid_vi 
        log_likelihood = self.log_likelhood(beta_values, start_points_batch, end_points_batch, step_times_batch)
        elogp = tf.reduce_sum(self.gh_grid_weights_vi*log_likelihood)
        penalty = kl_weight * \
            tfp.distributions.kl_divergence(
                self.variational_posterior, self.prior)

        return -elogp+penalty

    # fast vectorised tf code

    @tf.function
    def log_likelhood(self, beta_params, start_points, end_points, step_times):
        # calculate the covariates at the step end locations using tensorflow probability
        env_points = tf.math.mod(end_points, self.L)

        cov_xy_end = tfp.math.batch_interp_regular_nd_grid(
            env_points, self.x_ref_min, self.x_ref_max, self.cov_tensor, axis=-3)
        # calculate the rsf
        rsf_points_end = tf.exp(tf.reduce_sum(tf.multiply(tf.expand_dims(
            cov_xy_end, 0), tf.expand_dims(beta_params, 1)), axis=-1))

        # sigma_var = 1.0 ## just for now - needs to be optimised too
        sigmas = tf.math.sqrt(self.move_std * 0.5*step_times)

        def fn(input_locations):

            # shapes needs to broadcast with (num_steps,ghx points, ghy points, 2)
            half_sigma = tf.reshape(sigmas, (-1, 1, 1, 1))
            means = tf.reshape(input_locations, (-1, 1, 1, 2))

            # shape is numvisamples, num steps, ghx,ghy, 1xnc matrix of coefficients
            betas = tf.reshape(beta_params, (-1, 1, 1, 1, 1, self.n_covars))
            gh_points = tf.math.mod(
                means+(2**0.5)*half_sigma*self.gh_grid, self.L)
            cov_gh_points = tfp.math.batch_interp_regular_nd_grid(
                gh_points, self.x_ref_min, self.x_ref_max, self.cov_tensor, axis=-3)
            # need to add a dimension at the start to broadcast with num samples
            cov_gh_points = tf.expand_dims(
                tf.expand_dims(cov_gh_points, -1), 0)

            rsf = tf.exp(tf.matmul(betas, cov_gh_points)[..., 0, 0])

            # rsf is now dimension (numvisamples, numsteps, ghx, ghy)
            # multiply by the weights and sum to approximate the inner integral
            ysum = tf.reduce_sum(rsf*self.grid_gh_weights/(np.pi**0.5), axis=2)
            # multiply by the weights and sum to approximate the outer integral
            xsum = tf.reduce_sum(ysum*self.gh_weights/(np.pi**0.5), axis=2)
            return xsum

        half_sigma = tf.reshape(sigmas, (1, 1, -1, 1))

        # reshape to be ghx, ghx, num steps, 2
        mean_c = tf.reshape(0.5*(start_points+end_points), (1, 1, -1, 2))

        mid_points = mean_c + half_sigma*tf.expand_dims(self.gh_grid, 2)

        inv_int_z = tf.math.pow(tf.map_fn(fn, tf.reshape(
            mid_points, (self.n_gh_points*self.n_gh_points, -1, 2)), parallel_iterations=1), -1)

        inv_int_z = tf.reshape(inv_int_z, (self.n_gh_points, self.n_gh_points, tf.shape(
            inv_int_z)[1], tf.shape(inv_int_z)[2]))

        # inv_int_z is now shape (ghx,ghy,num vi samples, num steps)
        ysum = tf.reduce_sum(inv_int_z*tf.reshape(self.grid_gh_weights,
                             (self.n_gh_points, self.n_gh_points, 1, 1))/(np.pi**0.5), axis=0)
        xsum = tf.reduce_sum(
            ysum*tf.reshape(self.gh_weights, (self.n_gh_points, 1, 1))/(np.pi**0.5), axis=0)

        step_log_prob = tfp.distributions.Independent(tfp.distributions.Normal(
            loc=start_points, scale=(2**0.5)*sigmas), reinterpreted_batch_ndims=1).log_prob(end_points)

        log_prob = tf.math.log(rsf_points_end) + \
            tf.math.log(xsum) + step_log_prob
        return tf.math.reduce_sum(log_prob, axis=-1)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]
