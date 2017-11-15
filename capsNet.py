import tensorflow as tf
import tensorflow.contrib.layers as tcl

from config import cfg
from utils import get_batch_data
from capsLayer import CapsLayer


class CapsNet(object):

    def __init__(self, is_training=True):
        self.graph = tf.Graph()

        # input / output
        self.X = None
        self.labels = None
        self.Y = None

        self.v_length = None
        self.decoded = None
        self.argmax_idx = None

        # train
        self.global_step = None
        self.train_op = None

        # loss
        self.margin_loss = None
        self.reconstruction_loss = None
        self.total_loss = None

        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_batch_data()
                self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            elif cfg.mask_with_y:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
                self.Y = tf.placeholder(tf.float32, shape=(cfg.batch_size, 10, 1))
                self.build_arch()
            else:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
                self.build_arch()
            pass

        tf.logging.info('Seting up the main structure')
        pass

    def build_arch(self):
        # 做一层局部特征提取：使Capsule的输入和输出都是vector
        with tf.variable_scope('Conv1_layer'):
            # [?, 20, 20, 256]
            conv1 = tcl.conv2d(self.X, num_outputs=256, kernel_size=9, stride=1, padding='VALID')
            pass

        # 初始Capsule层：多个常规卷积层的堆叠，把8个conv2d拼接在一起，形成一个neural unit(capsule)
        # neural unit的输出为 8*1的vector
        # Primary Capsules layer, return [?, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)  # [batch_size, 1152, 8, 1]
            pass

        # DigitCaps layer, return [?, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)  # [?, 10, 16, 1]

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            epsilon = 1e-9
            # [?, 10, 16, 1] => [?, 10, 1, 1]
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)

            # b). pick out the index of max softmax val of the 10 caps
            # [?, 10, 1, 1] => [?, 1, 1] (index) => [?]
            argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            self.argmax_idx = tf.reshape(argmax_idx, shape=(cfg.batch_size, ))

            # Method 1.
            if not cfg.mask_with_y:
                # c). indexing
                # It's not easy to understand the indexing process with argmax_idx as we are 3-dim animal
                masked_v = []
                for which in range(cfg.batch_size):
                    # v Representation of the reconstruction target
                    v = self.caps2[which][self.argmax_idx[which], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))
                    pass
                self.masked_v = tf.concat(masked_v, axis=0)  # [?, 1, 16, 1]
            # Method 2. masking with true label, default mode
            else:
                # [?, 10, 16] x [?, 10, 1] => [?, 10, 16]
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
                pass

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [?, 1, 16, 1] => [?, 16] => [?, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))  # [?, 16]
            fc1 = tcl.fully_connected(vector_j, num_outputs=512)  # [?, 512]
            fc2 = tcl.fully_connected(fc1, num_outputs=1024)  # [?, 1024]
            self.decoded = tcl.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)  # [?, 784]
            pass

        pass

    def loss(self):
        # 1. The margin loss
        # max_l = max(0, m_plus-||v_c||)^2
        # max_r = max(0, ||v_c||-m_minus)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))  # [?, 10, 1, 1]
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))  # [?, 10, 1, 1]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))  # [?, 10, 1, 1] => [?, 10]
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))  # [?, 10, 1, 1] => [?, 10]

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        t_c = self.Y
        # [batch_size, 10], element-wise multiply
        l_c = t_c * max_l + cfg.lambda_val * (1 - t_c) * max_r
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(l_c, axis=1))

        # 2. The reconstruction loss
        orgin_x = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin_x)
        self.reconstruction_loss = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.total_loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_loss

        pass

    # Summary
    def _summary(self):
        train_summary = [tf.summary.scalar('train/margin_loss', self.margin_loss),
                         tf.summary.scalar('train/reconstruction_loss', self.reconstruction_loss),
                         tf.summary.scalar('train/total_loss', self.total_loss)]
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])

    pass
