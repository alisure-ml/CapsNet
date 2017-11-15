import tensorflow as tf
from tqdm import tqdm

from utils import load_mnist, save_images
from capsNet import CapsNet


class Runner:

    def __init__(self, batch_size, model_path="log"):
        self.batch_size = batch_size
        self.model_path = model_path

        # data
        self.num_train_batch = 60000 // self.batch_size
        self.num_test_batch = 10000 // self.batch_size
        self.test_x, self.test_y = load_mnist(is_training=False)

        # net
        self.capsNet = CapsNet(batch_size=self.batch_size, is_training=True, use_recons_loss=True, recon_with_y=True)

        # A training helper that checkpoints models and computes summaries.
        self.sv = tf.train.Supervisor(graph=self.capsNet.graph, logdir=model_path, save_model_secs=0)
        self.sess = self.sv.managed_session()
        pass

    def train(self, epochs=6, test_sum_freq=500, save_model_freq=3):
        for epoch in range(epochs):
            if self.sv.should_stop():
                break
            for step in tqdm(range(self.num_train_batch), total=self.num_train_batch, ncols=70, leave=False, unit='b'):
                # train
                _ = self.sess.run(self.capsNet.train_op)
                # test
                if (step + 1) % test_sum_freq == 0:
                    self.test(epoch)
            # save model
            if epoch % save_model_freq == 0:
                self.sv.saver.save(self.sess, self.model_path + '/model_epoch_%04d' % epoch)
        pass

    def recons(self, number=8, result_path="result"):
        number_2 = number * number
        if self.capsNet.recon_with_y:
            feed_dict = {self.capsNet.x: self.test_x[0: number_2], self.capsNet.labels: self.test_y[0: number_2]}
        else:
            feed_dict = {self.capsNet.x: self.test_x[0: number_2]}

        _, decoded = self.sess.run([self.capsNet.masked_v, self.capsNet.decoded], feed_dict=feed_dict)
        save_images(decoded, [number, number], path=result_path)
        pass

    # test
    def test(self, epoch):
        test_acc = 0
        for i in range(self.num_test_batch):
            start = i * self.batch_size
            end = start + self.batch_size
            test_acc += self.sess.run(self.capsNet.batch_accuracy, {self.capsNet.x: self.test_x[start:end],
                                                                    self.capsNet.labels: self.test_y[start:end]})
        test_acc = test_acc / (self.batch_size * self.num_test_batch)
        print("{} {}".format(epoch, test_acc))
        return test_acc

    pass

if __name__ == "__main__":
    runner = Runner(batch_size=128)
    runner.train()
    runner.recons()

