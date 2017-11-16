# encoding:UTF-8
import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np
from utils import load_mnist, save_images, save_txt
from capsNet import CapsNet


class Runner:

    def __init__(self, batch_size, use_recons_loss, recon_with_y, model_path="log"):
        self.batch_size = batch_size
        self.type_name = "with_y" if recon_with_y else "no_y"
        self.model_path = os.path.join(model_path, self.type_name)

        # data
        self.num_train = 60000 // self.batch_size
        self.num_test = 10000 // self.batch_size
        self.test_x, self.test_y = load_mnist(is_training=False)

        # net
        self.capsNet = CapsNet(batch_size=self.batch_size, use_recons_loss=use_recons_loss, recon_with_y=recon_with_y)

        # A training helper that checkpoints models and computes summaries.
        self.sv = tf.train.Supervisor(graph=self.capsNet.graph, logdir=self.model_path, save_model_secs=0)
        pass

    # 训练
    def train(self, epochs=10, test_freq=1, recons_freq=5, save_model_freq=1):
        with self.sv.managed_session() as sess:
            for epoch in range(epochs):
                # stop
                if self.sv.should_stop():
                    break
                # train
                for _ in tqdm(range(self.num_train), total=self.num_train, ncols=70, leave=False, unit='b'):
                    _ = sess.run(self.capsNet.train_op)
                # test
                if epoch % test_freq == 0:
                    self._test(sess, epoch)
                # recons
                if epoch % recons_freq == 0:
                    self._recons(sess, result_file="result/result_{}_{}.bmp".format(self.type_name, epoch))
                # save model
                if epoch % save_model_freq == 0:
                    self.sv.saver.save(sess, self.model_path + '/model_epoch_%04d' % epoch)
                pass
        pass

    # 1.重构
    def recons(self):
        with self.sv.managed_session() as sess:
            self._recons(sess, result_file="result/result_{}.bmp".format(self.type_name), recons_file="recons/mask.txt")
        pass

    # 重构
    def _recons(self, sess, result_file, recons_file):
        x = self.test_x[0: self.batch_size]
        y = self.test_y[0: self.batch_size]
        feed_dict = {self.capsNet.x: x, self.capsNet.labels: y} if self.capsNet.recon_with_y else {self.capsNet.x: x}

        masked_v, decoded = sess.run([self.capsNet.recons_input, self.capsNet.decoded], feed_dict=feed_dict)
        save_txt(masked_v, recons_file_name=recons_file)
        save_images(decoded, result_file_name=result_file, height_number=8)
        pass

    # 2.随机重构
    def recons_random(self, change_speed=0.3):
        baseline = [
            [-0.284, -0.254, 0.136, -0.224, 0.28, 0.271, 0.192, 0.328, -0.273, -0.189, -0.206, -0.0693, -0.317, -0.231, -0.278, 0.199],
            [0.278, -0.289, -0.216, 0.253, -0.189, 0.131, -0.265, -0.246, 0.3, 0.243, 0.343, -0.251, -0.156, 0.0995, 0.23, -0.253],
            [0.263, -0.161, -0.234, -0.176, -0.187, 0.336, -0.308, 0.254, 0.292, -0.268, -0.228, 0.222, -0.28, -0.168, 0.169, 0.253],
            [-0.196, 0.221, -0.231, -0.146, -0.213, 0.351, 0.184, 0.353, 0.189, -0.227, -0.213, 0.113, 0.206, -0.242, -0.267, -0.238],
            [0.262, -0.349, -0.23, 0.254, 0.207, -0.163, -0.223, -0.156, 0.271, 0.307, 0.295, -0.187, 0.212, -0.0903, 0.28, -0.228],
            [-0.261, 0.226, 0.233, -0.173, 0.181, 0.251, 0.174, 0.282, 0.126, -0.351, -0.19, -0.263, 0.263, 0.175, -0.223, 0.278],
            [0.297, 0.274, -0.139, -0.326, 0.312, -0.149, -0.374, 0.0468, 0.118, -0.299, 0.306, 0.137, -0.208, -0.277, -0.1, -0.138],
            [-0.281, 0.273, 0.198, 0.221, -0.34, -0.241, 0.188, 0.24, 0.135, -0.178, 0.254, -0.271, 0.195, -0.229, 0.313, 0.131],
            [-0.202, -0.13, -0.209, -0.191, -0.251, 0.121, -0.37, -0.331, -0.335, -0.277, -0.168, 0.251, -0.226, 0.26, -0.126, -0.165],
            [-0.206, -0.143, 0.193, -0.0517, -0.123, -0.0842, 0.134, -0.098, 0.184, 0.0991, 0.0826, 0.138, 0.119, 0.211, 0.212, -0.125]
        ]

        with self.sv.managed_session() as sess:
            for which_number in range(10):
                input_random = np.zeros(shape=[self.batch_size, 16], dtype=np.float32)
                for index in [0, 1]:
                    for row_index in range(8):
                        start_index = row_index * 8
                        input_random[start_index, :] = baseline[which_number]
                        for col_index in range(1, 8):
                            now_data = np.copy(baseline[which_number])
                            now_data[row_index + index * 8] = (col_index - 4) * change_speed
                            input_random[start_index + col_index, :] = now_data
                            pass
                        pass

                    decoded = sess.run(self.capsNet.decoded, feed_dict={self.capsNet.recons_input: input_random})
                    save_images(decoded, result_file_name="recons/random_{}_{}_{}.bmp".format(which_number, change_speed, index), height_number=8)
                pass

        pass

    # 3.随机重构:缓慢变化
    def recons_random_slow(self):
        baseline = [
            [-0.284, -0.254, 0.136, -0.224, 0.28, 0.271, 0.192, 0.328, -0.273, -0.189, -0.206, -0.0693, -0.317, -0.231, -0.278, 0.199],
            [0.278, -0.289, -0.216, 0.253, -0.189, 0.131, -0.265, -0.246, 0.3, 0.243, 0.343, -0.251, -0.156, 0.0995, 0.23, -0.253],
            [0.263, -0.161, -0.234, -0.176, -0.187, 0.336, -0.308, 0.254, 0.292, -0.268, -0.228, 0.222, -0.28, -0.168, 0.169, 0.253],
            [-0.196, 0.221, -0.231, -0.146, -0.213, 0.351, 0.184, 0.353, 0.189, -0.227, -0.213, 0.113, 0.206, -0.242, -0.267, -0.238],
            [0.262, -0.349, -0.23, 0.254, 0.207, -0.163, -0.223, -0.156, 0.271, 0.307, 0.295, -0.187, 0.212, -0.0903, 0.28, -0.228],
            [-0.261, 0.226, 0.233, -0.173, 0.181, 0.251, 0.174, 0.282, 0.126, -0.351, -0.19, -0.263, 0.263, 0.175, -0.223, 0.278],
            [0.297, 0.274, -0.139, -0.326, 0.312, -0.149, -0.374, 0.0468, 0.118, -0.299, 0.306, 0.137, -0.208, -0.277, -0.1, -0.138],
            [-0.281, 0.273, 0.198, 0.221, -0.34, -0.241, 0.188, 0.24, 0.135, -0.178, 0.254, -0.271, 0.195, -0.229, 0.313, 0.131],
            [-0.202, -0.13, -0.209, -0.191, -0.251, 0.121, -0.37, -0.331, -0.335, -0.277, -0.168, 0.251, -0.226, 0.26, -0.126, -0.165],
            [-0.206, -0.143, 0.193, -0.0517, -0.123, -0.0842, 0.134, -0.098, 0.184, 0.0991, 0.0826, 0.138, 0.119, 0.211, 0.212, -0.125]
        ]

        change_speed = (0.5 - -0.5) / 63
        with self.sv.managed_session() as sess:
            for number_index in range(10):
                decodes = []
                for attr_index in range(len(baseline[0])):
                    input_random = np.zeros(shape=[self.batch_size, 16], dtype=np.float32)
                    input_random[0, :] = baseline[number_index]
                    for col_index in range(1, self.batch_size):
                        now_data = np.copy(baseline[number_index])
                        now_data[attr_index] = (col_index - 32) * change_speed
                        input_random[col_index, :] = now_data
                        pass

                    decoded = sess.run(self.capsNet.decoded, feed_dict={self.capsNet.recons_input: input_random})
                    decodes.extend(decoded)
                    pass

                save_images(decodes, result_file_name="recons/random_{}_{:.4}.bmp".format(number_index, change_speed),
                            height_number=16)
            pass

        pass

    # 测试
    def test(self, info="test"):
        with self.sv.managed_session() as sess:
            self._test(sess, info)
        pass

    def _test(self, sess, info):
        test_acc = 0
        for i in range(self.num_test):
            start = i * self.batch_size
            end = start + self.batch_size
            test_acc += sess.run(self.capsNet.batch_accuracy, {self.capsNet.x: self.test_x[start:end],
                                                               self.capsNet.labels: self.test_y[start:end]})
        test_acc = test_acc / (self.batch_size * self.num_test)
        print("{} {}".format(info, test_acc))
        return test_acc

    pass

if __name__ == "__main__":
    runner = Runner(batch_size=64, use_recons_loss=True, recon_with_y=False)
    # 训练
    runner.train()
    # 测试
    runner.test()
    # 重构结果
    runner.recons()
    # 低级的随机重构
    runner.recons_random()
    # 高级的随机重构
    runner.recons_random_slow()

