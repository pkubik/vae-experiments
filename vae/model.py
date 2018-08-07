from collections import namedtuple
from pathlib import Path
from typing import Union

import tensorflow as tf

from vae.data import Dataset
from vae.image import norm_images
from vae.net import VAE
from vae.plot import PlotSaverHook


class Model:
    def __init__(self, model_dir: Union[str, Path]):
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.config_proto = tf.ConfigProto(gpu_options=self.gpu_options)
        self.steps_before_eval = 1000
        self.batch_size = 128
        self.num_epochs = None
        self.image_shape = (28, 28, 1)
        self.model_dir = Path(model_dir)

    def create_data_iterators(self, data: Dataset):
        images = norm_images(data.images)
        labels = data.labels

        return tf.estimator.inputs.numpy_input_fn(
            x=images,
            y=labels,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            shuffle=True)()

    def train(self, data: Dataset):
        graph = tf.Graph()
        with graph.as_default():
            global_step = tf.train.get_or_create_global_step()

            images_iterator, labels_iterator = self.create_data_iterators(data)

            vae = VAE(images_iterator, labels_iterator)
            vae_train = vae.create_train_spec()

            plot_saver_hook = PlotSaverHook(self.model_dir / 'plots', vae, self.steps_before_eval)

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=str(self.model_dir),
                    config=self.config_proto,
                    hooks=[plot_saver_hook]) as sess:
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

                    progbar.update(i, [
                        ('rec loss', outputs.rec_loss),
                        ('reg loss', outputs.reg_loss),
                        ('t scale', outputs.t_scale_mean)])
