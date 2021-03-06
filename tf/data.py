import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
# enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)


class FLAGS(object):
    def __init__(self):
        self.batch_size = 64 # "The number of batch images [64]")
        self.n_epoch = 50 # "Epoch to train [25]"

        self.lr = 2e-4 # "Learning rate of for adam [0.0002]")
        self.beta1 = 0.5 # "Momentum term of adam [0.5]")

        self.z_dim = 100  # "Num of noise value]"
        self.output_size = 28  # "The size of the output images to produce [64]")
        self.sample_size = 64 # "The number of sample images [64]")
        self.c_dim = 1  # "Number of image channels. [3]")

        self.save_every_epoch = 1 # "The interval of saveing checkpoints.")
        self.checkpoint_dir = "checkpoint" # "Directory name to save the checkpoints [checkpoint]")
        self.sample_dir = "samples" # "Directory name to save the image samples [samples]")
        assert np.sqrt(self.sample_size) % 1 == 0., 'Flag `sample_size` needs to be a perfect square'


flags = FLAGS()

tl.files.exists_or_mkdir(flags.checkpoint_dir) # save model
tl.files.exists_or_mkdir(flags.sample_dir) # save generated image


def get_mnist(batch_size):
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
    train_set = X_train
    print(train_set.shape)
    length = len(train_set)

    def generator_train():
        for img in train_set:
            yield (img.reshape([28, 28, 1]) - 0.5) / 0.5 # a Tensor with values range in [-1, 1]

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.float32)
    ds = train_ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    return ds, length


if __name__ == '__main__':

    images, len_instance = get_mnist(flags.output_size, flags.n_epoch, flags.batch_size)
    for step, batch_images in enumerate(images):
        if step != 0:
            break
        print(batch_images.shape)
        tl.visualize.save_images(batch_images.numpy(), [8, 8], 'show.png')