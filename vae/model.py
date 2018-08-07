from collections import namedtuple

from abc import ABC
from pathlib import Path
from typing import Union

import tensorflow as tf
import tensorpack as tp
import tensorpack.tfutils.summary

from vae.data import Dataset
from vae.image import norm_images
from vae.net import VAE
from vae.plot import PlotSaverHook


# noinspection PyAttributeOutsideInit
class ModelDesc(tp.ModelDesc, ABC):
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def inputs(self):
        """
        Define all the inputs (with type, shape, name) that the graph will need.
        """
        return [tf.placeholder(tf.float32, (None, *self.image_shape), 'image'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, image, label):
        """This function should build the model which takes the input variables
        and return cost at the end"""

        image = tf.expand_dims(image, -1)
        self.vae = VAE(image, label)
        self.vae_train = self.vae.create_train_spec()

        return self.vae_train.vlb.total_loss

    def optimizer(self):
        return self.vae_train.optimizer


class Model:
    def __init__(self, model_dir: Union[str, Path]):
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.config_proto = tf.ConfigProto(gpu_options=self.gpu_options)
        self.steps_before_eval = 1000
        self.batch_size = 128
        self.num_epochs = 10
        self.image_shape = (28, 28)
        self.model_dir = Path(model_dir)
        self.model_desc = ModelDesc(self.image_shape)

    def create_data_flows(self):
        train = tp.BatchData(tp.dataset.Mnist('train'), self.batch_size)
        test = tp.BatchData(tp.dataset.Mnist('test'), self.batch_size, remainder=True)
        return train, test

    def train(self):
        tp.logger.set_logger_dir(str(self.model_dir), 'k')
        dataset_train, dataset_test = self.create_data_flows()

        config = tp.AutoResumeTrainConfig(
            model=self.model_desc,
            data=tp.QueueInput(dataset_train),
            callbacks=[
                tp.ModelSaver(checkpoint_dir=str(self.model_dir)),
                tp.MinSaver('validation_rec_loss'),
                tp.InferenceRunner(
                    dataset_test,
                    [tp.ScalarStats(['rec_loss'])]),
                tp.ProgressBar(['rec_loss'])
            ],
            max_epoch=self.num_epochs,
        )
        tp.launch_train_with_config(config, tp.SimpleTrainer())
