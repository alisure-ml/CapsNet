import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl

from config import cfg


class CapsLayer(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.

    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type
        pass

    def __call__(self, input, kernel_size=None, stride=None):
        """
        The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
        """
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            # the PrimaryCaps layer, a convolutional layer
            # input: [?, 20, 20, 256]
            if not self.with_routing:

                '''
                # version 1, computational expensive
                capsules = []
                for i in range(self.vec_len):
                    # each capsule i: [batch_size, 6, 6, 32]
                    with tf.variable_scope('ConvUnit_' + str(i)):
                        caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                                                          self.kernel_size, self.stride,
                                                          padding="VALID", activation_fn=None)
                        caps_i = tf.reshape(caps_i, shape=(cfg.batch_size, -1, 1, 1))
                        capsules.append(caps_i)
                assert capsules[0].get_shape() == [cfg.batch_size, 1152, 1, 1]
                capsules = tf.concat(capsules, axis=2)
                '''

                # version 2, equivalent to version 1 but higher computational efficiency.
                # NOTE: I can't find out any words from the paper whether the
                # PrimaryCap convolution does a ReLU activation or not before
                # squashing function, but experiment show that using ReLU get a
                # higher test accuracy. So, which one to use will be your choice
                capsules = tcl.conv2d(input, self.num_outputs * self.vec_len, self.kernel_size, self.stride,
                                      padding="VALID", activation_fn=tf.nn.relu)  # [?, 256, 6, 6]
                capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))  # [?, 1152, 8, 1]
                return self.squash(capsules)  # [?, 1152, 8, 1]

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [?, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))

                with tf.variable_scope('routing'):
                    capsules = self.routing(self.input)  # [?, 1, 10, 16, 1]
                    capsules = tf.squeeze(capsules, axis=1)
                return capsules
            pass
        pass

    def routing(self, input):
        """ 
        The routing algorithm.
        Args:
            input: A Tensor with [batch_size, num_caps_in=1152, 1, length(u_i)=8, 1] shape, 
            num_caps_l meaning the number of capsule in the layer l.
        Returns:
            A Tensor of shape [batch_size, num_caps_out, length(v_j)=16, 1] representing 
            the vector output `v_j` in the layer l+1
        Notes:
            u_i represents the vector output of capsule i in the layer in, and
            v_j the vector output of capsule j in the layer out.
         """

        # Eq.2, u_hat：W_ij * u_i
        # W: [num_caps_j, num_caps_i, len_u_i, len_v_j]
        w = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=cfg.stddev))
        # do tiling for input and W before matmul
        input = tf.tile(input, [1, 1, 10, 1, 1])  # input => [?, 1152, 10, 8, 1]
        w = tf.tile(w, [cfg.batch_size, 1, 1, 1, 1])  # W => [?, 1152, 10, 8, 16]
        u_hat = tf.matmul(w, input, transpose_a=True)  # [8, 16].T x [8, 1] => [16, 1] => [?, 1152, 10, 16, 1]

        # line 2:
        # b_IJ: [batch_size, num_caps_in, num_caps_out, 1, 1],
        # about the reason of using 'batch_size', see issue #21
        b_ij = np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32)
        b_ij = tf.constant(b_ij)  # line 2: b_ij <- 0  # [batch_size, 1152, 10, 1, 1]
        v_j = None

        # line 3:
        # for r iterations do
        for r_iter in range(cfg.iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                # line 4: b_ij.shape = [?, 1152, 10, 1, 1]
                c_ij = tf.nn.softmax(b_ij, dim=2)  # [?, 1152, 10, 1, 1]

                # line 5:
                # weighting u_hat with c_ij, element-wise in the last two dims
                # [?, 1152, 10, 1, 1] x [?, 1152, 10, 16, 1] => [?, 1152, 10, 16, 1]
                s_j = tf.multiply(c_ij, u_hat)  # [?, 1152, 10, 16, 1]
                # then sum in the second dim, resulting in [?, 1, 10, 16, 1]
                s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)  # [?, 1, 10, 16, 1]

                # line 6:
                # squash using Eq.1,
                v_j = self.squash(s_j)  # [?, 1, 10, 16, 1]

                # line 7:
                v_j_tiled = tf.tile(v_j, [1, 1152, 1, 1, 1])  # [?, 1152, 10, 16, 1]
                # [?, 1152, 10, 16, 1].T  x [?, 1152, 10, 16, 1] => [?, 1152, 10, 1, 1]
                b_ij += tf.matmul(u_hat, v_j_tiled, transpose_a=True)
                pass
            pass
        return v_j

    # 挤压函数：向量单位化和缩放操作
    @staticmethod
    def squash(vector):
        """
        Squashing function corresponding to Eq.1
        Args:
            vector: A 5-D tensor with shape [batch_size, 1, num_caps, vec_len, 1],
        Returns:
            A 5-D tensor with the same shape as vector but squashed in 4rd and 5th dimensions.
        """
        epsilon = 1e-9
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = (vec_squared_norm / (1 + vec_squared_norm)) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = scalar_factor * vector  # element-wise
        return vec_squashed

    pass
