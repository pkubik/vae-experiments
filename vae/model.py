from collections import namedtuple

from pathlib import Path
from typing import Union

import tensorflow as tf

from vae import configurator as cfg
from vae.data import Dataset, save_config
from vae.image import norm_images
from vae.net.vi import VAE
from vae.plot import PlotSaverHook


class ProgressPrinterHook(tf.train.SessionRunHook):
    Fetches = namedtuple('Fetches', 'step, rec_loss, reg_loss')

    def __init__(self, steps_before_eval, rec_loss, reg_loss):
        self.step = tf.train.get_global_step()
        self.steps_before_eval = steps_before_eval
        self.rec_loss = rec_loss
        self.reg_loss = reg_loss
        self.progbar = tf.keras.utils.Progbar(
            self.steps_before_eval,
            stateful_metrics=['rec loss', 'reg loss'])

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            self.Fetches(self.step, self.rec_loss, self.reg_loss))

    def after_run(self, run_context, run_values):
        fetches = run_values.results  # type: ProgressPrinterHook.Fetches
        i = fetches.step % self.steps_before_eval
        if i == 0:
            self.progbar = tf.keras.utils.Progbar(
                self.steps_before_eval,
                stateful_metrics=['rec loss', 'reg loss'])
            print()
        else:
            self.progbar.update(i, [
                ('rec loss', fetches.rec_loss),
                ('reg loss', fetches.reg_loss)])


def create_input_fn(batch_size: int, data: Dataset):
    images = norm_images(data.images)
    labels = data.labels

    input_fn = tf.estimator.inputs.numpy_input_fn(
        images, labels,
        batch_size, 1,
        shuffle=True,
        queue_capacity=cfg.get('shuffle_buffer_size', 2048)
    )

    return input_fn


class Model:
    def __init__(self, model_dir: Union[str, Path]):
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.get('gpu_memory_fraction', 0.8))
        self.config_proto = tf.ConfigProto(gpu_options=self.gpu_options)
        self.steps_before_eval = cfg.get('steps_before_eval', 1000)
        self.batch_size = cfg.get('batch_size', 128)
        self.num_epochs = cfg.get('num_epochs', 100)
        self.image_shape = cfg.get('image_shape', [28, 28])
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def build(self, features, labels, mode):
        features = tf.expand_dims(features, -1)
        vae = VAE(features, labels)
        vae_train = vae.create_train_spec()

        plot_saver_hook = PlotSaverHook(self.model_dir / 'plots', vae, self.steps_before_eval)
        progress_printer_hook = ProgressPrinterHook(
            self.steps_before_eval,
            vae_train.vlb.reconstruction_loss,
            vae_train.vlb.regularization_loss)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=vae_train.vlb.total_loss,
            train_op=vae_train.train_op,
            training_chief_hooks=[plot_saver_hook, progress_printer_hook])

    def train(self, data: Dataset):
        estimator = tf.estimator.Estimator(self.build, self.model_dir)
        save_config(cfg.current(), self.model_dir / 'config.yml')

        for epoch in range(self.num_epochs):
            print("Starting epoch {}".format(epoch))
            estimator.train(create_input_fn(self.batch_size, data))
            print()
