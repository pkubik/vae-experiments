from pathlib import Path
from typing import Union

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.figure
import matplotlib.backends.backend_svg

from vae.data import Dataset
from vae.image import norm_images
from vae.net import VAE


def save_grid_plot(sess: tf.train.MonitoredSession, vae: VAE, path: Path):
    a = np.arange(-2., 2. + 10e-8, 0.2)
    t_grid = np.transpose(np.meshgrid(a, a))
    full_t_grid = np.pad(np.reshape(t_grid, [-1, 2]), ((0, 0), (0, vae.latent_dim - 2)), 'constant')

    images = sess.run_step_fn(
        lambda step_context: step_context.session.run(vae.x_mean, feed_dict={vae.t: full_t_grid}))

    plot_data = np.transpose(
        np.reshape(images, [len(a), len(a), 28, 28]),
        [0, 2, 1, 3])
    plot_data = plot_data.reshape([len(a) * 28, len(a) * 28])

    fig = mpl.figure.Figure()
    subplot = fig.add_subplot(111)
    subplot.imshow(plot_data)
    fig.axes[0].set_axis_off()

    path.mkdir(exist_ok=True)
    global_step = sess.run_step_fn(
        lambda step_context: step_context.session.run(tf.train.get_global_step()))
    plot_path = path / 'grid_{}.svg'.format(global_step)

    mpl.backends.backend_svg.FigureCanvas(fig)
    fig.savefig(str(plot_path))


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

        graph = tf.Graph()
        with graph.as_default():
            tf.train.get_or_create_global_step()

            batch_iterator = tf.estimator.inputs.numpy_input_fn(
                x=images,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                shuffle=True)()

            vae = VAE(batch_iterator)
            vae_train = vae.create_train_spec()

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=str(self.model_dir),
                    config=self.config_proto) as sess:

                progbar = tf.keras.utils.Progbar(self.steps_before_eval)
                i = 0
                while not sess.should_stop():
                    _, loss, t, t_scale_mean, rec_loss, reg_loss, x_batch, x_mean_batch = sess.run(
                        [vae_train.train_op,
                         vae_train.vlb.total_loss,
                         vae.t,
                         vae.t_scale_mean,
                         vae_train.vlb.reconstruction_loss,
                         vae_train.vlb.regularization_loss,
                         vae.x,
                         vae.x_mean])

                    i += 1
                    if i == self.steps_before_eval:
                        i = 1
                        progbar = tf.keras.utils.Progbar(self.steps_before_eval)
                        save_grid_plot(sess, vae, self.model_dir / 'plots')

                    progbar.update(i, [
                        ('rec loss', rec_loss),
                        ('reg loss', reg_loss),
                        ('t scale', t_scale_mean)])

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
