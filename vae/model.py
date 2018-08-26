from collections import namedtuple

from contextlib import suppress
from pathlib import Path
from typing import Union

import tensorflow as tf

from vae import configurator as cfg
from vae.data import Dataset, save_config
from vae.image import norm_images
from vae.net import VAE
from vae.plot import PlotSaverHook


class DataIterator:
    def __init__(self, batch_size: int, data: Dataset):
        images = norm_images(data.images)
        labels = data.labels
        self.data = Dataset(images, labels)

        self.images_placeholder = tf.placeholder(images.dtype, images.shape)
        self.labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

        dataset = tf.data.Dataset.from_tensor_slices(
            Dataset(self.images_placeholder, self.labels_placeholder))
        self.iterator = (dataset
                         .shuffle(cfg.get('shuffle_buffer_size', 2048))
                         .batch(batch_size)
                         .repeat(1)
                         .map(lambda x: Dataset(tf.expand_dims(x.images, -1), x.labels))
                         .make_initializable_iterator())
        self.next = self.iterator.get_next()

    def initialize(self, sess: tf.Session):
        sess.run(self.iterator.initializer, feed_dict={
            self.images_placeholder: self.data.images,
            self.labels_placeholder: self.data.labels
        })


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

    def train(self, data: Dataset):
        graph = tf.Graph()
        with graph.as_default():
            global_step = tf.train.get_or_create_global_step()

            data_iterator = DataIterator(self.batch_size, data)

            vae = VAE(data_iterator.next.images, data_iterator.next.labels)
            vae_train = vae.create_train_spec()

            plot_saver_hook = PlotSaverHook(self.model_dir / 'plots', vae, self.steps_before_eval)

            save_config(cfg.current(), self.model_dir / 'config.yml')

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=str(self.model_dir),
                    config=self.config_proto,
                    hooks=[plot_saver_hook]) as sess:
                for epoch in range(self.num_epochs):
                    print("Starting epoch {}".format(epoch))

                    progbar = tf.keras.utils.Progbar(self.steps_before_eval)

                    with suppress(tf.errors.OutOfRangeError):
                        data_iterator.initialize(sess._tf_sess())

                        while not sess.should_stop():
                            Fetches = namedtuple('Fetches',
                                                 'global_step,'
                                                 'train_op, rec_loss, reg_loss')
                            fetches = Fetches(global_step,
                                              vae_train.train_op,
                                              vae_train.vlb.reconstruction_loss,
                                              vae_train.vlb.regularization_loss)
                            outputs = sess.run(fetches)

                            i = outputs.global_step % self.steps_before_eval
                            if i == 0:
                                progbar = tf.keras.utils.Progbar(self.steps_before_eval)

                            progbar.update(i, [
                                ('rec loss', outputs.rec_loss),
                                ('reg loss', outputs.reg_loss)])

                    print()
