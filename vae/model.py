from abc import ABC
from pathlib import Path
from typing import Union

import tensorflow as tf
import tensorpack as tp

from vae.data import Dataset, load_mnist
from vae.image import norm_images
from vae.net import VAE


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


class Model(tp.Trainer):
    def __init__(self, model_dir: Union[str, Path]):
        super().__init__()
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.config_proto = tf.ConfigProto(gpu_options=self.gpu_options)
        self.steps_before_eval = 1000
        self.batch_size = 128
        self.num_epochs = 10
        self.image_shape = (28, 28)
        self.model_dir = Path(model_dir)

        tp.logger.set_logger_dir(str(self.model_dir), 'k')
        images, labels = self.create_data_iterators(load_mnist().train)
        self.vae = VAE(images, labels)
        self.train_spec = self.vae.create_train_spec()
        self.train_op = self.train_spec.train_op
        self.steps_per_epoch = 400

    def create_data_iterators(self, data: Dataset):
        images = norm_images(data.images)
        labels = data.labels

        return tf.estimator.inputs.numpy_input_fn(
            x=images,
            y=labels,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            shuffle=True)()

    #
    # def train(self):
    #     tp.logger.set_logger_dir(str(self.model_dir), 'k')
    #     dataset_train, dataset_test = self.create_data_flows()
    #
    #     config = tp.AutoResumeTrainConfig(
    #         model=self.model_desc,
    #         data=tp.QueueInput(dataset_train),
    #         callbacks=[
    #             tp.ModelSaver(checkpoint_dir=str(self.model_dir)),
    #             tp.MinSaver('validation_rec_loss'),
    #             tp.InferenceRunner(
    #                 dataset_test,
    #                 [tp.ScalarStats(['rec_loss'])])
    #         ],
    #         max_epoch=self.num_epochs,
    #     )
    #     tp.launch_train_with_config(config, tp.SimpleTrainer())
