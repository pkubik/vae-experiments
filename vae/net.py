from collections import namedtuple

import tensorflow as tf
from vae import configurator as cfg


def clip(x, eps=10e-16):
    return tf.clip_by_value(x, eps, 1 - eps)


class VLB:
    def __init__(self, x, x_decoded_mean, t_mean, t_log_var):
        """Variational Lower Bound for Gaussian `p(x | t)`.

        Inputs:
            x: (batch_size x width x height x num_channels)
                tensor of the input images
            x_decoded_mean: (batch_size x width x height x num_channels)
                mean of the estimated distribution `p(x | t)`, real numbers from 0 to 1
            t_mean: (batch_size x latent_dim)
                mean vector of the (normal) distribution `q(t | x)`
            t_log_var: (batch_size x latent_dim)
                logarithm of the variance vector of the (normal) distribution `q(t | x)`

        Returns:
            A tf.Tensor with one element (averaged across the batch), VLB
        """
        batch_size = tf.shape(x)[0]

        # Reconstruction loss, log p(x | t)
        flat_x = tf.reshape(x, [batch_size, -1])
        flat_x_mean = tf.reshape(x_decoded_mean, [batch_size, -1])
        x_mse = tf.reduce_sum(tf.square(flat_x - flat_x_mean), -1)
        image_var = cfg.get('image_dist_var', 1 / 4)
        rec_loss = x_mse / 2 / image_var  # Assuming sigma of x equals 1/2
        self.reconstruction_loss = tf.reduce_mean(rec_loss)

        # Regularization loss, KL(q || p)
        t_dist = tf.distributions.Normal(t_mean, tf.exp(t_log_var / 2))
        t_prior = tf.distributions.Normal(tf.zeros_like(t_mean), tf.ones_like(t_mean))
        kl_t = tf.reduce_sum(tf.distributions.kl_divergence(t_dist, t_prior), -1)
        self.regularization_loss = tf.reduce_mean(kl_t)

        self.total_loss = self.reconstruction_loss + self.regularization_loss

        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
        tf.summary.scalar('regularization_loss', self.regularization_loss)


class Encoder:
    def __init__(self, latent_dim, num_labels):
        scale = cfg.get('encoder_scale', 5)
        self.conv_layers = [
            tf.layers.Conv2D(scale, 3, padding='SAME', activation=tf.nn.relu),
            tf.layers.Conv2D(scale * 2, 2, activation=tf.nn.relu),
            tf.layers.Conv2D(scale * 4, 3, 2, activation=tf.nn.relu),
            tf.layers.Conv2D(scale * 10, 3, activation=tf.nn.relu),
            tf.layers.Conv2D(scale * 10, 3, 2, activation=tf.nn.relu),
            tf.layers.Conv2D(scale * 20, 3, activation=tf.nn.relu)
        ]
        self.mid_layer = tf.layers.Dense(scale * 40, activation=tf.tanh)
        self.mean_layer = tf.layers.Dense(latent_dim)
        self.log_var_layer = tf.layers.Dense(latent_dim)
        self.label_layer = tf.layers.Dense(num_labels)

    Output = namedtuple('EncoderOutput', 't_dist, label_logits')

    def __call__(self, x) -> Output:
        """
        Generate parameters for the estimated `q(t | x)` distribution.
        """
        h = x
        for conv in self.conv_layers:
            h = conv(h)

        final_h = self.mid_layer(tf.layers.flatten(h))

        mean = self.mean_layer(final_h)
        std = tf.exp(self.log_var_layer(final_h) / 2)
        label = self.label_layer(final_h)
        return self.Output(tf.distributions.Normal(mean, std), label)


class Decoder:
    def __init__(self):
        scale = cfg.get('decoder_scale', 10)
        self.scale = scale
        self.init_layer = tf.layers.Dense(scale * 90, activation=tf.nn.relu)

        self.conv_layers = [
            tf.layers.Conv2DTranspose(scale * 10, 3, activation=tf.nn.relu),
            tf.layers.Conv2DTranspose(scale * 5, 3, 2, activation=tf.nn.relu),
            tf.layers.Conv2DTranspose(scale * 5, 3, activation=tf.nn.relu),
            tf.layers.Conv2DTranspose(scale * 2, 3, 2, activation=tf.nn.relu),
            tf.layers.Conv2DTranspose(scale * 2, 2, activation=tf.nn.relu),
            tf.layers.Conv2DTranspose(scale, 3)
        ]

    def __call__(self, t):
        """
        Output the image distribution `p(x | t)` given the latent code.
        """
        init_h = self.init_layer(t)

        h = tf.reshape(init_h, [-1, 3, 3, self.scale * 10])
        for conv in self.conv_layers:
            h = conv(h)

        h = tf.layers.conv2d(h, 1, 3)

        return h


VAETrainSpec = namedtuple('VAETrainSpec', 'train_op, vlb, loss')


class VAE:
    def __init__(self, x, label, num_labels=10):
        self.latent_dim = cfg.get('latent_dim', 2)
        self.cond_latent_dim = cfg.get('cond_latent_dim', 10)
        self.class_emb_dim = cfg.get('class_emb_dim', 5)

        self.image_shape = list(x.shape[1:])
        self.num_labels = num_labels
        self.x = tf.placeholder_with_default(x, (None, *self.image_shape), name='x')
        self.label = tf.placeholder_with_default(label, (None,), name='label')
        self.encoder = Encoder(self.latent_dim, self.num_labels)
        self.decoder = Decoder()

        # Compute `q(t | x)` parameters, feed prior p(t) to hallucinate
        self.t_dist, self.label_logits = self.encoder(self.x)
        self.t_scale_mean = tf.reduce_mean(self.t_dist.scale)

        # Sample `t`
        self.t = tf.identity(self.t_dist.sample(), name='t')

        # Generate mean output distribution `p(x | t)`
        self.label_embedding = tf.layers.dense(
            tf.one_hot(self.label, self.num_labels),
            self.class_emb_dim,
            name='class_emb')
        self.t_layer = tf.layers.Dense(self.cond_latent_dim, name='cond_t')
        self.augmented_t = self.t_layer(tf.concat([self.t, self.label_embedding], -1))
        self.x_mean = tf.identity(self.decoder(self.augmented_t), name='x_mean')

        tf.summary.histogram('t0', self.t[0])
        tf.summary.histogram('t_std', tf.nn.moments(self.t, -1)[1])
        tf.summary.histogram('labels', tf.argmax(self.label_logits, -1))
        tf.summary.histogram('target_labels', self.label)
        tf.summary.image('x_mean', self.x_mean)

    def create_train_spec(self):
        learning_rate = cfg.get('learning_rate', 0.001)
        vlb = VLB(self.x, self.x_mean, self.t_dist.loc, tf.log(self.t_dist.scale) * 2)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(vlb.total_loss, tf.train.get_global_step())

        return VAETrainSpec(train_op, vlb, vlb.total_loss)
