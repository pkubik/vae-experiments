from collections import namedtuple
from pathlib import Path
from typing import Union

import tensorflow as tf
import numpy as np

from vae.data import Dataset
from vae.image import norm_images
from vae.net import VAE


class Model:
    def __init__(self, model_dir: Union[str, Path]):
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.config_proto = tf.ConfigProto(gpu_options=self.gpu_options)
        self.steps_before_eval = 200
        self.batch_size = 64
        self.num_epochs = None
        self.image_shape = (28, 28, 1)
        self.model_dir = Path(model_dir)

    def train(self, data: Dataset):

        images = norm_images(data.images)

        graph = tf.Graph()
        with graph.as_default():
            tf.train.get_or_create_global_step()

            # dataset = tf.data.Dataset.from_tensor_slices(images)
            # dataset = dataset.repeat(self.num_epochs)
            # dataset = dataset.shuffle(1)
            # dataset = dataset.batch(self.batch_size)

            # batch_iterator = dataset.make_one_shot_iterator().get_next()
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
                    if i == progbar.target:
                        progbar.target *= 2
                        # image, test_t = sess.run([vae.x_mean, vae.t], feed_dict={
                        #     vae.x: [data[1], np.expand_dims(subset[-1], -1)],
                        #     vae.t_dist.scale: np.zeros([1, vae.latent_dim])
                        # })
                        # sampled_t, sampled_image = sess.run([vae.t, vae.x_mean], feed_dict={
                        #     vae.t_dist.loc: np.zeros([self.batch_size, vae.latent_dim]),
                        #     vae.t_dist.scale: np.ones([self.batch_size, vae.latent_dim])
                        # })

                        # clear_output(wait=True)
                        # fig = plt.figure(figsize=(12, 16))
                        # grid = plt.GridSpec(3, 2)
                        #
                        # plt.subplot(grid[0, 0], title='Train example')
                        # plt.imshow(data[1, :, :, 0])
                        #
                        # plt.subplot(grid[0, 1], title='Train reconstruction')
                        # plt.imshow(np.clip(image[0, :, :, 0], 0., 1.))
                        #
                        # plt.subplot(grid[1, 0], title='Test example')
                        # plt.imshow(subset[-1])
                        #
                        # plt.subplot(grid[1, 1], title='Test reconstruction')
                        # plt.imshow(np.clip(image[1, :, :, 0], 0., 1.))
                        #
                        # plt.subplot(grid[2, 0], title='t values (first 2 dims)')
                        # x_values = np.reshape(x_batch, [-1])
                        # plt.scatter(t[:, 0], t[:, 1])
                        # plt.scatter(test_t[0, 0], test_t[0, 1], s=64)
                        # plt.scatter(test_t[1, 0], test_t[1, 1], s=64, marker='x')
                        # plt.scatter(sampled_t[0, 0], sampled_t[0, 1], s=64, marker='^')
                        #
                        # plt.subplot(grid[2, 1], title='Sampled image')
                        # plt.imshow(np.clip(sampled_image[0, :, :, 0], 0., 1.))
                        #
                        # plt.show()
                        #
                        # self._plot_grid(sess, vae)

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
