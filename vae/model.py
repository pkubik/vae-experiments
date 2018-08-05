from collections import namedtuple
from pathlib import Path
from typing import Union

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from vae.data import Dataset
from vae.image import norm_images
from vae.net import VAE


def raw_run(sess: tf.train.MonitoredSession, fetches, feed_dict):
    """
    Runs the `tf.train.MonitoredSession` without the session hooks.

    :param sess: session to run
    :param fetches: fetches (see `tf.Session.run`)
    :param feed_dict: feed dict (see `tf.Session.run`)
    :return: fetched outputs
    """
    def step_fn(step_context):
        return step_context.session.run(fetches, feed_dict)
    return sess.run_step_fn(step_fn)


def save_grid_plot(sess: tf.train.MonitoredSession, label: int, step: int, vae: VAE, path: Path):
    """
    Plots images generated using latent codes that form a grid on the first
    2 dimensions of the latent space.

    Currently, due to introduction of the label condition, only digit 5 is plotted.

    :param sess: managed session to use for drawing predictions
    :param step: step count which will used in the filename
    :param vae: `VAE` object used to construct the `sess` graph
    :param path: path to the directory with plots
    """
    a = np.arange(-2., 2. + 10e-8, 0.2)
    t_grid = np.transpose(np.meshgrid(a, a))
    full_t_grid = np.pad(np.reshape(t_grid, [-1, 2]), ((0, 0), (0, vae.latent_dim - 2)), 'constant')

    images = raw_run(sess, vae.x_mean, feed_dict={
        vae.t: full_t_grid,
        vae.label: np.ones(full_t_grid.shape[0]) * label
    })

    plot_data = np.transpose(
        np.reshape(images, [len(a), len(a), 28, 28]),
        [0, 2, 1, 3])
    plot_data = plot_data.reshape([len(a) * 28, len(a) * 28])

    path.mkdir(exist_ok=True)
    plot_path = path / 'grid_{}_{}.png'.format(label, step)

    plt.imsave(str(plot_path), plot_data, format="png")


def save_grid_plots(sess: tf.train.MonitoredSession, step: int, vae: VAE, path: Path):
    for i in range(10):
        save_grid_plot(sess, i, step, vae, path)


class Model:
    def __init__(self, model_dir: Union[str, Path]):
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.config_proto = tf.ConfigProto(gpu_options=self.gpu_options)
        self.steps_before_eval = 1000
        self.batch_size = 128
        self.num_epochs = None
        self.image_shape = (28, 28, 1)
        self.model_dir = Path(model_dir)

    def train(self, data: Dataset):

        images = norm_images(data.images)
        labels = data.labels

        graph = tf.Graph()
        with graph.as_default():
            global_step = tf.train.get_or_create_global_step()

            images_iterator, labels_iterator = tf.estimator.inputs.numpy_input_fn(
                x=images,
                y=labels,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                shuffle=True)()

            vae = VAE(images_iterator, labels_iterator)
            vae_train = vae.create_train_spec()

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=str(self.model_dir),
                    config=self.config_proto) as sess:

                progbar = tf.keras.utils.Progbar(self.steps_before_eval)
                while not sess.should_stop():
                    Fetches = namedtuple('Fetches',
                                         'global_step,'
                                         'train_op, total_loss, t, t_scale_mean,'
                                         'rec_loss, reg_loss, x, x_mean')
                    fetches = Fetches(global_step,
                                      vae_train.train_op,
                                      vae_train.vlb.total_loss,
                                      vae.t,
                                      vae.t_scale_mean,
                                      vae_train.vlb.reconstruction_loss,
                                      vae_train.vlb.regularization_loss,
                                      vae.x,
                                      vae.x_mean)
                    outputs = sess.run(fetches)

                    i = outputs.global_step % self.steps_before_eval
                    if i == 0:
                        progbar = tf.keras.utils.Progbar(self.steps_before_eval)
                        save_grid_plots(sess, outputs.global_step, vae, self.model_dir / 'plots')

                    progbar.update(i, [
                        ('rec loss', outputs.rec_loss),
                        ('reg loss', outputs.reg_loss),
                        ('t scale', outputs.t_scale_mean)])

    def generate_from_latent(self, t):
        graph = tf.Graph()
        with graph.as_default():
            tf.train.get_or_create_global_step()

            x_shape = (None, *self.image_shape, 1)
            x = tf.placeholder(tf.float32, x_shape, 'x_placeholder')

            vae = VAE(x)

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=str(self.model_dir),
                    config=self.config_proto,
                    save_checkpoint_steps=0,
                    save_summaries_steps=0) as sess:
                images = sess.run(vae.x_mean, feed_dict={vae.t: np.array(t)})
                return images
