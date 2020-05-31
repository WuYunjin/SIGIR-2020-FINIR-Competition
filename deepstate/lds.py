# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from ..utils import make_nd_diag, _broadcast_param,MultivariateGaussian,Gaussian
from ..utils.func import jitter_cholesky, mxnet_swapaxes
# print('emission , innovation ,transition ,noise_std,residuals 维度如下：'
#               ,self.emission_coeff[0].shape , self.innovation_coeff[0].shape , self.transition_coeff[0].shape , self.noise_std[0].shape , self.residuals[0].shape)

class LDS(object):
    r"""
    Implements Linear Dynamical System (LDS) as a distribution.

    The LDS is given by

    .. math::
        z_t = A_t l_{t-1} + b_t + \epsilon_t \\
        l_t = C_t l_{t-1} + g_t \nu

    where

    .. math::
        \epsilon_t = N(0, S_v) \\
        \nu = N(0, 1)

    :math:`A_t`, :math:`C_t` and :math:`g_t` are the emission, transition and
    innovation coefficients respectively. The residual terms are denoted
    by :math:`b_t`.

    The target :math:`z_t` can be :math:`d`-dimensional in which case

    .. math::
        A_t \in R^{d \times h}, b_t \in R^{d}, C_t \in R^{h \times h}, g_t \in R^{h}

    where :math:`h` is dimension of the latent state.

    Parameters
    ----------
    emission_coeff
        Tensor of shape (batch_size, seq_length, obs_dim, latent_dim)
    transition_coeff
        Tensor of shape (batch_size, seq_length, latent_dim, latent_dim)
    innovation_coeff
        Tensor of shape (batch_size, seq_length, latent_dim)
    noise_std
        Tensor of shape (batch_size, seq_length, obs_dim)
    residuals
        Tensor of shape (batch_size, seq_length, obs_dim)
    prior_mean
        Tensor of shape (batch_size, latent_dim)
    prior_cov
        Tensor of shape (batch_size, latent_dim, latent_dim)
    latent_dim
        Dimension of the latent state
    output_dim
        Dimension of the output
    seq_length
        Sequence length
    F
    """

    def __init__(
        self,
        emission_coeff,
        transition_coeff,
        innovation_coeff,
        noise_std,
        residuals,
        prior_mean,
        prior_cov,
        latent_dim,
        output_dim,
        seq_length,
    ) -> None:
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seq_length = seq_length

        self.emission_coeff= tf.transpose(emission_coeff , [1,0,2,3])#(336, 32, 1, 32)

        self.innovation_coeff = tf.expand_dims(
            tf.transpose(innovation_coeff , [1,0,2])
            ,axis=2
        )#(336, 32, 1, 32)

        self.transition_coeff = tf.transpose(transition_coeff , [1,0,2,3])#(336, 32, 32, 32)

        self.noise_std = tf.transpose(noise_std , [1,0,2])#(336, 32, 1)

        self.residuals = tf.transpose(residuals, [1, 0, 2])#(336, 32, 1)

        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

    @property
    def batch_shape(self) :
        return tuple(self.emission_coeff[0].shape[:1]) + (self.seq_length,)

    @property
    def event_shape(self) :
        return (self.output_dim,)

    @property
    def event_dim(self) -> int:
        return 2

    def log_prob(
        self,
        x,
        scale = None,
        observed = None,
    ):
        """
        给定观测值以及 协变量，计算 序列之间的log_likelihood
        ----------
        x
            Observations, shape (batch_size, seq_length, output_dim)
        scale
            Scale of each sequence in x, shape (batch_size, output_dim)
        observed
            Flag tensor indicating which observations are genuine (1.0) and
            which are missing (0.0)

        Returns
        -------
        Tensor
            Log probabilities, shape (batch_size, seq_length)
        Tensor
            Final mean, shape (batch_size, latent_dim)
        Tensor
            Final covariance, shape (batch_size, latent_dim, latent_dim)
        """
        # 对 x进行 缩放
        if scale is not None:
            x = tf.math.divide(x,tf.expand_dims(scale,axis=1))

        log_p, final_mean, final_cov ,all_pred_mean , all_pred_cov = self.kalman_filter(x, observed)
        return log_p, final_mean, final_cov , all_pred_mean , all_pred_cov

    def kalman_filter(
        self, targets, observed
    ) :
        """
        计算 target
        Parameters
        ----------
        targets
            Observations, shape (batch_size, seq_length, output_dim)
        observed
            Flag tensor indicating which observations are genuine (1.0) and
            which are missing (0.0)

        Returns
        -------
        Tensor
            Log probabilities, shape (batch_size, seq_length)
        Tensor
            Mean of p(l_T | l_{T-1}), where T is seq_length, with shape
            (batch_size, latent_dim)
        Tensor
            Covariance of p(l_T | l_{T-1}), where T is seq_length, with shape
            (batch_size, latent_dim, latent_dim)
        """
        # targets[t]: (batch_size, obs_dim)
        # targets = [tf.squeeze(tensor,axis=1)
        #     for tensor in tf.split(targets ,num_or_size_splits=self.seq_length ,axis=1)]
        targets = tf.transpose(targets , [1,0,2])

        mean = self.prior_mean
        cov = self.prior_cov

        observed = (
            # [tf.squeeze(tensor, axis=1)
            #     for tensor in tf.split(observed, num_or_size_splits=self.seq_length, axis=1)]
            tf.transpose(observed , [1,0])
            if observed is not None
            else None
        )

        t = 0
        log_p_seq = tf.TensorArray(size=self.seq_length ,dtype=tf.float32)
        all_pred_mu = tf.TensorArray(size=self.seq_length , dtype=tf.float32)
        all_pred_cov = tf.TensorArray(size=self.seq_length, dtype=tf.float32)

        def cond(t, mean, cov, log_p_seq , all_pred_mu  ,all_pred_cov):
            return tf.less(t, self.seq_length)

        def body(t, mean, cov, log_p_seq , all_pred_mu  ,all_pred_cov):
            filtered_mean, filtered_cov, log_p = kalman_filter_step(
                target=targets[t],
                prior_mean=mean,
                prior_cov=cov,
                emission_coeff=self.emission_coeff[t],
                residual=self.residuals[t],
                noise_std=self.noise_std[t],
                latent_dim=self.latent_dim,
                output_dim=self.output_dim,
            )

            # log_p_seq.append(tf.expand_dims(log_p, axis=1))
            log_p_seq = log_p_seq.write(t,tf.expand_dims(log_p,axis=1))

            # Mean of p(l_{t+1} | l_t)
            mean = tf.squeeze(tf.linalg.matmul(
                    self.transition_coeff[t],
                    (
                        tf.expand_dims(filtered_mean, axis=-1)
                        if observed is None
                        else tf.expand_dims(tf.where(
                                tf.math.equal(observed[t], 1), x=filtered_mean, y=mean
                            ), axis=-1
                        )
                    )
                ), axis=-1
            )

            all_pred_mu = all_pred_mu.write(t , mean)

            # Covariance of p(l_{t+1} | l_t)
            cov = tf.linalg.matmul(
                self.transition_coeff[t],
                tf.linalg.matmul(
                    (
                        filtered_cov
                        if observed is None
                        else tf.where(tf.math.equal(observed[t], 1), x=filtered_cov, y=cov)
                    ),
                    self.transition_coeff[t],
                    transpose_b=True,
                ),
            ) + tf.linalg.matmul(
                self.innovation_coeff[t],
                self.innovation_coeff[t],
                transpose_a=True,
            )

            all_pred_cov = all_pred_cov.write(t, cov)

            t = t + 1

            return t, mean, cov, log_p_seq ,all_pred_mu ,all_pred_cov

        _, mean, cov, log_p_seq ,all_pred_mu ,all_pred_cov = tf.while_loop(cond, body,
                                                loop_vars=[t, mean, cov, log_p_seq , all_pred_mu, all_pred_cov])
        # print('log_p_seq stack shape' , log_p_seq.stack().shape)
        # exit()

        log_p_seq = tf.transpose(
            tf.squeeze(log_p_seq.stack() ,axis=-1)
            , [1,0]
        )
        all_pred_mu = tf.transpose(all_pred_mu.stack() , [1,0,2]) #(bs ,seq ,dim_l)
        all_pred_cov = tf.transpose(all_pred_cov.stack() , [1,0,2,3]) #(bs,seq , dim_l)

        # Return sequence of log likelihoods, as well as
        # final mean and covariance of p(l_T | l_{T-1} where T is seq_length
        return log_p_seq, mean, cov, all_pred_mu, all_pred_cov

    def sample(
        self, num_samples = None, scale = None
    ) :
        r"""
        Generates samples from the LDS: p(z_1, z_2, \ldots, z_{`seq_length`}).

        Parameters
        ----------
        num_samples
            Number of samples to generate
        scale
            Scale of each sequence in x, shape (batch_size, output_dim)

        Returns
        -------
        Tensor
            Samples, shape (num_samples, batch_size, seq_length, output_dim)
        """

        # Note on shapes: here we work with tensors of the following shape
        # in each time step t: (num_samples, batch_size, dim, dim),
        # where dim can be obs_dim or latent_dim or a constant 1 to facilitate
        # generalized matrix multiplication (gemm2)

        # Sample observation noise for all time steps
        # noise_std: (batch_size, seq_length, obs_dim, 1)
        noise_std = tf.expand_dims(
            tf.transpose(self.noise_std , [1,0,2])
            ,axis=-1
        ) #(bs,pred_len , obs_dim ,1)

        # samples_eps_obs[t]: (num_samples, batch_size, obs_dim, 1)
        samples_eps_obs = (
            tf.transpose(
                Gaussian(tf.zeros_like(noise_std), noise_std).sample(num_samples),
                [2,0,1,3,4]
            )
        ) #(pred_len, num_sample, bs, obs_dim, 1)


        # Sample standard normal for all time steps
        # samples_eps_std_normal[t]: (num_samples, batch_size, obs_dim, 1)
        samples_std_normal = (
            tf.transpose(
                Gaussian(tf.zeros_like(noise_std), tf.ones_like(noise_std)).sample(num_samples)
                ,[2,0,1,3,4]
            )
        )
        print('tensorflow : samples_eps_obs, samples_std_normal的维度 ：'
              , samples_eps_obs.shape
              , samples_std_normal.shape
        )

        # Sample the prior state.
        # samples_lat_state: (num_samples, batch_size, latent_dim, 1)
        # The prior covariance is observed to be slightly negative definite whenever there is
        # excessive zero padding at the beginning of the time series.
        # We add positive tolerance to the diagonal to avoid numerical issues.
        # Note that `jitter_cholesky` adds positive tolerance only if the decomposition without jitter fails.
        state = MultivariateGaussian(
            self.prior_mean, #(bs, latent_dim)
            jitter_cholesky(
                self.prior_cov, self.latent_dim, float_type=np.float32
            ), #(bs, latent_dim , latent_dim)
        )
        samples_lat_state = tf.expand_dims(
            state.sample(num_samples)
            ,axis=-1
        )#(num_sample, bs, latent_dim,1)
        t=0
        samples_seq = tf.TensorArray(size=self.seq_length , dtype=tf.float32)

        def cond(t,samples_lat_state,samples_seq):
            return tf.less(t, self.seq_length)
        def body(t,samples_lat_state,samples_seq):
            # Expand all coefficients to include samples in axis 0
            # emission_coeff_t: (num_samples, batch_size, obs_dim, latent_dim)
            # transition_coeff_t:
            #   (num_samples, batch_size, latent_dim, latent_dim)
            # innovation_coeff_t: (num_samples, batch_size, 1, latent_dim)
            emission_coeff_t, transition_coeff_t, innovation_coeff_t = [
                _broadcast_param(coeff, axes=[0], sizes=[num_samples])
                if num_samples is not None
                else coeff
                for coeff in [
                    self.emission_coeff[t],
                    self.transition_coeff[t],
                    self.innovation_coeff[t],
                ]
            ]  # (num_sample, bs, obs, latent) (num_sample, bs, latent, latent) (num_sample, bs, 1, latent)

            # Expand residuals as well
            # residual_t: (num_samples, batch_size, obs_dim, 1)
            residual_t = (
                _broadcast_param(
                    tf.expand_dims(self.residuals[t], axis=-1),
                    axes=[0],
                    sizes=[num_samples],
                )
                if num_samples is not None
                else tf.expand_dims(self.residuals[t], axis=-1)
            )

            # (num_samples, batch_size, 1, obs_dim)
            samples_t = (
                    tf.linalg.matmul(emission_coeff_t, samples_lat_state)
                    + residual_t
                    + samples_eps_obs[t]
            )

            samples_t = (
                mxnet_swapaxes(samples_t, dim1=2, dim2=3)
                if num_samples is not None
                else mxnet_swapaxes(samples_t, dim1=1, dim2=2)
            )

            samples_seq = samples_seq.write(t ,samples_t)



            # sample next state: (num_samples, batch_size, latent_dim, 1)
            samples_lat_state = tf.linalg.matmul(
                transition_coeff_t, samples_lat_state #(samples,bs,lat,lat)×(samples, bs, lat,1 )
            ) + tf.linalg.matmul(
                innovation_coeff_t, samples_std_normal[t], transpose_a=True
            ) #(samples, bs, lat_dim,1 )×(samples, bs, obs_dim, 1) ->(samples,bs,lat,1)

            return t+1,samples_lat_state,samples_seq

        _,_,samples_seq = tf.while_loop(cond, body ,loop_vars=[t,samples_lat_state,samples_seq])


        #(?,num_samples, batch_size, 1, obs_dim)

        # 这里需要修改成 while_loop
        # (num_samples, batch_size, seq_length, obs_dim)
        samples = tf.transpose(tf.squeeze(samples_seq.stack(),axis=-2),[1,2,0,3])

        return (
            samples
            if scale is None
            else tf.math.multiply(
                samples,
                tf.expand_dims(tf.expand_dims(scale,axis=1),axis=0)
                if num_samples is not None
                else scale.expand_dims(axis=1),
            )
        )# (num_samples, batch_size, seq_length, obs_dim)

    def sample_marginals(
        self, num_samples = None, scale = None
    ) -> Tensor:
        r"""
        Generates samples from the marginals p(z_t),
        t = 1, \ldots, `seq_length`.

        Parameters
        ----------
        num_samples
            Number of samples to generate
        scale
            Scale of each sequence in x, shape (batch_size, output_dim)

        Returns
        -------
        Tensor
            Samples, shape (num_samples, batch_size, seq_length, output_dim)
        """

        state_mean = tf.expand_dims(self.prior_mean,axis=-1)
        state_cov = self.prior_cov

        output_mean_seq = []
        output_cov_seq = []

        for t in range(self.seq_length):
            # compute and store observation mean at time t
            output_mean = tf.linalg.matmul(
                self.emission_coeff[t], state_mean
            ) + self.residuals[t].expand_dims(axis=-1)

            output_mean_seq.append(output_mean)

            # compute and store observation cov at time t
            output_cov = tf.linalg.matmul(
                self.emission_coeff[t],
                tf.linalg.matmul(
                    state_cov, self.emission_coeff[t], transpose_b=True
                ),
            ) + make_nd_diag(
                x=self.noise_std[t] * self.noise_std[t], d=self.output_dim
            )

            output_cov_seq.append(tf.expand_dims(output_cov,axis=1))

            state_mean = tf.linalg.matmul(self.transition_coeff[t], state_mean)

            state_cov = tf.linalg.matmul(
                self.transition_coeff[t],
                tf.linalg.matmul(
                    state_cov, self.transition_coeff[t], transpose_b=True
                ),
            ) + tf.linalg.matmul(
                self.innovation_coeff[t],
                self.innovation_coeff[t],
                transpose_a=True,
            )

        output_mean = tf.concat(output_mean_seq, axis=1)
        output_cov = tf.concat(output_cov_seq, axis=1)

        L = tf.linalg.cholesky(output_cov)

        output_distribution = MultivariateGaussian(output_mean, L)

        samples = output_distribution.sample(num_samples=num_samples)

        return (
            samples
            if scale is None
            else tf.math.multiply(samples, tf.expand_dims(scale,axis=1))
        )


class LDSArgsProj(object):
    def __init__(
        self,
        output_dim: int,
        noise_std_ub: float = 1.0,
        innovation_ub: float = 0.01,
    ) -> None:
        super().__init__()
        with tf.variable_scope('lstm_proj'):
            self.output_dim = output_dim
            self.dense_noise_std = tf.layers.Dense(
                units=1,
                activation=tf.keras.activations.softrelu
                if noise_std_ub is float("inf")
                else tf.keras.activations.softplus ,
                name = 'noise_std'
            )
            self.dense_innovation = tf.layers.Dense(
                units=1,
                activation=tf.keras.activations.softplus
                if innovation_ub is float("inf")
                else tf.keras.activations.sigmoid,
                name = 'innovation'
            )
            self.dense_residual = tf.layers.Dense(
                units=output_dim,
                name='residual'
            )

            self.innovation_factor = (
                1.0 if innovation_ub is float("inf") else innovation_ub
            )
            self.noise_factor = (
                1.0 if noise_std_ub is float("inf") else noise_std_ub
            )

    # noinspection PyMethodOverriding,PyPep8Naming
    # 返回了 Tuple[Tensor, Tensor, Tensor]
    # x Input的内容是lstm 的输出结果 #(bs , seq_length , hidden_state)
    def build_forward(self, x) :
        noise_std = self.dense_noise_std(x) * self.noise_factor
        innovation = self.dense_innovation(x) * self.innovation_factor
        residual = self.dense_residual(x)

        return noise_std, innovation, residual


def  kalman_filter_step(
    target: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
    emission_coeff: Tensor,
    residual: Tensor,
    noise_std: Tensor,
    latent_dim: int,
    output_dim: int,
):
    """
    One step of the Kalman filter.

    This function computes the filtered state (mean and covariance) given the
    linear system coefficients the prior state (mean and variance),
    as well as observations.

    Parameters
    ----------
    target
        Observations of the system output, shape (batch_size, output_dim)
    prior_mean
        Prior mean of the latent state, shape (batch_size, latent_dim)
    x
        Prior covariance of the latent state, shape
        (batch_size, latent_dim, latent_dim)
    emission_coeff
        Emission coefficient, shape (batch_size, output_dim, latent_dim)
    residual
        Residual component, shape (batch_size, output_dim)
    noise_std
        Standard deviation of the output noise, shape (batch_size, output_dim)
    latent_dim
        Dimension of the latent state vector
    Returns
    -------
    Tensor
        Filtered_mean, shape (batch_size, latent_dim)
    Tensor
        Filtered_covariance, shape (batch_size, latent_dim, latent_dim)
    Tensor
        Log probability, shape (batch_size, )
    """
    # output_mean: mean of the target (batch_size, obs_dim)
    output_mean = tf.squeeze(tf.matmul(emission_coeff, tf.expand_dims(prior_mean, axis=-1)), axis=-1)

    # 计算 A
    # noise covariance(batch_size , output_dim , output_dim)
    noise_cov = make_nd_diag(x=noise_std * noise_std, d=output_dim)
    #(batch_size , latent_dim , output_dim)
    S_hh_x_A_tr = tf.linalg.matmul(prior_cov, emission_coeff, transpose_b=True)
    # covariance of the target  (batch_size ,obs_dim ,obs_dim)
    output_cov = tf.linalg.matmul(emission_coeff, S_hh_x_A_tr) + noise_cov

    # compute the Cholesky decomposition output_cov = LL^T
    L_output_cov = tf.linalg.cholesky(output_cov)

    # Compute Kalman gain matrix K:
    # K = S_hh X with X = A^T output_cov^{-1}
    # We have X = A^T output_cov^{-1} => X output_cov = A^T => X LL^T = A^T
    # We can thus obtain X by solving two linear systems involving L
    kalman_gain = tf.transpose(tf.linalg.triangular_solve(
        tf.transpose(L_output_cov, [0,2,1]),
        tf.linalg.triangular_solve(
            L_output_cov, tf.transpose(S_hh_x_A_tr,[0,2,1])
        ),
    ) , [0,2,1])


    # compute the error
    target_minus_residual = target - residual
    delta = target_minus_residual - output_mean

    # filtered estimates
    filtered_mean = tf.expand_dims(prior_mean,axis=-1) + tf.linalg.matmul(
        kalman_gain, tf.expand_dims(delta,axis=-1)
    )
    # 由于目标序列都是 univariate 的 (bs , latent_dim)
    filtered_mean = tf.squeeze(filtered_mean,axis=-1)

    # Joseph's symmetrized update for covariance:
    ImKA = tf.math.subtract(
        tf.eye(latent_dim), tf.linalg.matmul(kalman_gain, emission_coeff)
    )

    # 与 书本中Kalman滤波的推导不太一致 ,可以保留
    # 但是与 Kalan VAE 中的推导是一致的 (bs , latent_dim ,latent_dim)
    filtered_cov = tf.linalg.matmul(
        ImKA, tf.linalg.matmul(prior_cov, ImKA, transpose_b=True)
    ) + tf.linalg.matmul(
        kalman_gain, tf.linalg.matmul(noise_cov, kalman_gain, transpose_b=True)
    )

    # likelihood term: (batch_size,)
    log_p = MultivariateGaussian(output_mean, L_output_cov).log_prob(
        target_minus_residual
    )

    return filtered_mean, filtered_cov, log_p
