# encoding:UTF-8
import tensorflow as tf
from tqdm import tqdm
import os
from utils import load_mnist, save_images
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
    def train(self, epochs=100, test_freq=5, recons_freq=5, save_model_freq=5):
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

    # 重构
    def recons(self):
        with self.sv.managed_session() as sess:
            self._recons(sess, result_file="result/result_{}.bmp".format(self.type_name))
        pass

    def _recons(self, sess, result_file):
        x = self.test_x[0: self.batch_size]
        y = self.test_y[0: self.batch_size]
        feed_dict = {self.capsNet.x: x, self.capsNet.labels: y} if self.capsNet.recon_with_y else {self.capsNet.x: x}

        masked_v, decoded = sess.run([self.capsNet.recons_input, self.capsNet.decoded], feed_dict=feed_dict)
        save_images(decoded, result_file_name=result_file, height_number=8)
        pass

    # TODO: 随机重构:还没有写好
    def recons_random(self):
        with self.sv.managed_session() as sess:
            random_input = []
            decoded = sess.run(self.capsNet.decoded, feed_dict={self.capsNet.recons_input: random_input})
            save_images(decoded, result_file_name="result/result_random_{}.bmp".format(self.type_name), height_number=8)
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
    runner = Runner(batch_size=64, use_recons_loss=True, recon_with_y=True)
    runner.train()
    runner.test()
    runner.recons()
