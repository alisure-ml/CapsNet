import os
import scipy.misc as misc
import numpy as np
import tensorflow as tf


def load_mnist(is_training, data_set="data/mnist"):
    fd = open(os.path.join(data_set, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_set, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_y = loaded[8:].reshape((60000, )).astype(np.int32)

    fd = open(os.path.join(data_set, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_set, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_y = loaded[8:].reshape((10000, )).astype(np.int32)

    train_x = tf.convert_to_tensor(train_x / 255., tf.float32)

    if is_training:
        return train_x, train_y
    else:
        return test_x / 255., test_y
    pass


def get_batch_data(batch_size, num_threads=8):
    train_x, train_y = load_mnist(is_training=True)
    data_queues = tf.train.slice_input_producer([train_x, train_y])
    x, y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size, capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32, allow_smaller_final_batch=False)
    return x, y


def save_images(images, size, path):
    images = (images + 1.) / 2  # inverse_transform
    return misc.imsave(path, _merge_images(images, size))


def _merge_images(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image
        pass
    return imgs


if __name__ == '__main__':
    X, Y = load_mnist(is_training=True)
    print(X.get_shape())
    print(X.dtype)
    x, y = get_batch_data(batch_size=128)
    print(x.get_shape())
    print(y.get_shape())

