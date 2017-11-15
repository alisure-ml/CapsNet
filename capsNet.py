import tensorflow as tf
import tensorflow.contrib.layers as tcl

from utils import get_batch_data
from capsLayer import CapsLayer


class CapsNet(object):

    # is_training=True 表示训练，可以选择是否使用重构损失(use_recons_loss)和重构时是否使用标签(recon_with_y)
    # is_training=False 表示不训练，可以选择重构时是否使用标签(recon_with_y)
    def __init__(self, batch_size, is_training, use_recons_loss, recon_with_y):
        self.graph = tf.Graph()

        # param
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_recons_loss = use_recons_loss
        self.recon_with_y = recon_with_y

        # input
        self.x, self.labels = self._x_labels(self.is_training, self.batch_size)

        # network: result + reconstruction
        self.caps_digit, self.v_length, self.prediction, self.batch_accuracy, self.masked_v, self.decoded = self.net()

        # loss
        self.margin_loss, _, self.total_loss = self.loss_total(self.x, self.labels, self.v_length, self.decoded)

        # train
        self.train_op = self._train_op(self.use_recons_loss, self.margin_loss, self.total_loss)
        pass

    @staticmethod
    def _train_op(use_recons_loss, margin_loss, total_loss):
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(total_loss) if use_recons_loss else optimizer.minimize(margin_loss)

    @staticmethod
    def _x_labels(is_training, batch_size):
        x = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
        labels = tf.placeholder(tf.float32, shape=(batch_size, 10, 1))
        return get_batch_data(batch_size) if is_training else x, labels

    # 简单的网络：没有重构的部分
    def _build_simple_caps_net(self):
        # 做一层局部特征提取：使Capsule的输入和输出都是vector
        with tf.variable_scope('Conv1_layer'):
            # [?, 20, 20, 256]
            conv1 = tcl.conv2d(self.x, num_outputs=256, kernel_size=9, stride=1, padding='VALID')
            pass

        # 初始Capsule层：多个常规卷积层的堆叠，把8个conv2d拼接在一起，形成一个neural unit(capsule)
        # neural unit的输出为 8*1的vector
        # Primary Capsules layer, return [?, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primary_caps = CapsLayer(self.batch_size, num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps_primary = primary_caps(conv1, kernel_size=9, stride=2)  # [batch_size, 1152, 8, 1]
            pass

        # DigitCaps layer, return [?, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digit_caps = CapsLayer(self.batch_size, num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            caps_digit = digit_caps(caps_primary)  # [?, 10, 16, 1]

        with tf.variable_scope("Caps_prediction"):
            # calc ||v_c||, then do softmax(||v_c||)
            # [?, 10, 16, 1] => [?, 10, 1, 1]
            epsilon = 1e-9
            v_length = tf.sqrt(tf.reduce_sum(tf.square(caps_digit), axis=2, keep_dims=True) + epsilon)
            # [?, 10, 1, 1] => [?, 1, 1] (index) => [?]
            prediction = tf.to_int32(tf.argmax(tf.nn.softmax(v_length, dim=1), axis=1))
            prediction = tf.reshape(prediction, shape=(self.batch_size,))

        correct_prediction = tf.equal(tf.to_int32(self.labels), prediction)
        batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

        return caps_digit, v_length, prediction, batch_accuracy

    # 简单的网络+重构的部分
    def net(self):
        caps_digit, v_length, prediction, batch_accuracy = self._build_simple_caps_net()

        # Decoder structure in Fig. 2
        with tf.variable_scope('Masking'):
            masked_v = []
            if self.recon_with_y:  # 根据label取网络的输出
                # [?, 10, 16] x [?, 10, 1] => [?, 10, 16]
                y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)
                masked_v = tf.multiply(tf.squeeze(self.caps_digit), tf.reshape(y, (-1, 10, 1)))
            else:  # 取网络中最可能的输出
                for which in range(self.batch_size):
                    # v Representation of the reconstruction target
                    v = self.caps_digit[which][self.prediction[which], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))
                    pass
                masked_v = tf.concat(masked_v, axis=0)  # [?, 1, 16, 1]
                pass

        # reconstructe the MNIST images with 3 FC layers
        with tf.variable_scope('Decoder'):  # [?, 1, 16, 1] => [?, 16] => [?, 512]
            vector_j = tf.reshape(self.masked_v, shape=(self.batch_size, -1))  # [?, 16]
            fc1 = tcl.fully_connected(vector_j, num_outputs=512)  # [?, 512]
            fc2 = tcl.fully_connected(fc1, num_outputs=1024)  # [?, 1024]
            decoded = tcl.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)  # [?, 784]
            pass

        return caps_digit, v_length, prediction, batch_accuracy, masked_v, decoded

    # 1. The margin loss
    @staticmethod
    def _loss_margin(batch_size, v_length, labels, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
        y = tf.one_hot(labels, depth=10, axis=1, dtype=tf.float32)
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., m_plus - v_length))  # [?, 10, 1, 1]
        max_l = tf.reshape(max_l, shape=(batch_size, -1))  # [?, 10, 1, 1] => [?, 10]
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., v_length - m_minus))  # [?, 10, 1, 1]
        max_r = tf.reshape(max_r, shape=(batch_size, -1))  # [?, 10, 1, 1] => [?, 10]

        # [batch_size, 10], element-wise multiply
        l_c = y * max_l + lambda_val * (1 - y) * max_r
        return tf.reduce_mean(tf.reduce_sum(l_c, axis=1))

    # 2. The reconstruction loss
    @staticmethod
    def _loss_recons(batch_size, x, decoded):
        squared = tf.square(decoded - tf.reshape(x, shape=(batch_size, -1)))
        return tf.reduce_mean(squared)

    # 3. Total loss
    def loss_total(self, x, labels, v_length, decoded, regular_scale=0.0005*784):
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        margin_loss = self._loss_margin(self.batch_size, v_length, labels)
        recons_loss = self._loss_recons(self.batch_size, x, decoded)
        return margin_loss, recons_loss, margin_loss + regular_scale * recons_loss

    pass
