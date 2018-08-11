import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from vae.net import VAE


def save_grid_plot(sess: tf.Session, label: int, step: int, vae: VAE, path: Path):
    """
    Plots images generated using latent codes that form a grid on the first
    2 dimensions of the latent space.

    :param sess: managed session to use for drawing predictions
    :param label: label to be used when sampling
    :param step: step count which will used in the filename
    :param vae: `VAE` object used to construct the `sess` graph
    :param path: path to the directory with plots
    """
    a = np.arange(-2., 2. + 10e-8, 0.4)
    t_grid = np.transpose(np.meshgrid(a, a))
    full_t_grid = np.pad(np.reshape(t_grid, [-1, 2]), ((0, 0), (0, vae.latent_dim - 2)), 'constant')

    images = sess.run(vae.x_mean, feed_dict={
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


def save_grid_plots(sess: tf.Session, step: int, vae: VAE, path: Path):
    for i in [1, 4, 8]:  # 3 most interesting digits
        save_grid_plot(sess, i, step, vae, path)


def save_t_plot(t: np.ndarray, step: int, path: Path):
    plt.scatter(t[:, 0], t[:, 1])
    plt.savefig(str(path / "t_{}.svg".format(step)))
    plt.close()


class PlotSaverHook(tf.train.SessionRunHook):
    """Saves grid plots every N steps."""

    def __init__(self, path: Path, vae: VAE, steps_per_save=2000):
        self.path = path
        self.vae = vae
        self.steps_per_save = steps_per_save

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs({
            "step": tf.train.get_global_step(),
            "t_mean": self.vae.t_dist.loc
        })

    def after_run(self,
                  run_context: tf.train.SessionRunContext,
                  run_values: tf.train.SessionRunValues):
        step = run_values.results['step']
        t_mean = run_values.results['t_mean']
        if step % self.steps_per_save == 0:
            save_grid_plots(run_context.session, step, self.vae, self.path)
            save_t_plot(t_mean, step, self.path)
