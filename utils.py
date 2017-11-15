# encoding:UTF-8
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


def save_images(images, result_file_name, height_number):
    path = os.path.split(result_file_name)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    images = np.reshape(images, newshape=[len(images), 28, 28])
    return misc.imsave(result_file_name, _merge_images(images, height_number))


def _merge_images(images, height_number):
    number = len(images)
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * height_number, w * (number // height_number)))
    for x in range(height_number):
        for y in range(number//height_number):
            imgs[h * x: h * (x + 1), h * y: h * (y + 1)] = images[x * height_number + y, :]
        pass
    return imgs


if __name__ == '__main__':
    X, Y = load_mnist(is_training=True)
    print(X.get_shape())
    print(X.dtype)
    x, y = get_batch_data(batch_size=128)
    print(x.get_shape())
    print(y.get_shape())

