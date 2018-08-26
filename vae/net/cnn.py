import tensorflow as tf
from vae import configurator as cfg


class Encoder:
    def __init__(self, output_dim: int = None):
        if output_dim is None:
            output_dim = cfg.get('encoder_output_dim', 200)
        scale = cfg.get('encoder_scale', 5)
        self.conv_layers = [
            tf.layers.Conv2D(scale, 3, padding='SAME', activation=tf.nn.relu),
            tf.layers.Conv2D(scale * 2, 2, activation=tf.nn.relu),
            tf.layers.Conv2D(scale * 4, 3, 2, activation=tf.nn.relu),
            tf.layers.Conv2D(scale * 10, 3, activation=tf.nn.relu),
            tf.layers.Conv2D(scale * 10, 3, 2, activation=tf.nn.relu),
            tf.layers.Conv2D(scale * 20, 3, activation=tf.nn.relu)
        ]
        self.final_layer = tf.layers.Dense(output_dim, activation=tf.tanh)

    def __call__(self, x) -> tf.Tensor:
        """
        Generate parameters for the estimated `q(t | x)` distribution.
        """
        h = x
        for conv in self.conv_layers:
            h = conv(h)

        final_h = self.final_layer(tf.layers.flatten(h))
        return final_h


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
