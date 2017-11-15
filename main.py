import tensorflow as tf
from tqdm import tqdm

from config import cfg
from utils import load_mnist
from capsNet import CapsNet


class Runner:

    def run(self):
        # net
        capsNet = CapsNet(is_training=cfg.is_training)
        # A training helper that checkpoints models and computes summaries.
        sv = tf.train.Supervisor(graph=capsNet.graph, logdir=cfg.logdir, save_model_secs=0)
        with sv.managed_session() as sess:
            num_batch = 60000 // cfg.batch_size
            num_test_batch = 10000 // cfg.batch_size
            # data
            test_x, test_y = load_mnist(False)
            for epoch in range(cfg.epoch):
                global_step = None
                if sv.should_stop():
                    break
                for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                    # train
                    global_step, _ = sess.run([capsNet.global_step, capsNet.train_op])
                    # summary
                    if step % cfg.train_sum_freq == 0:
                        summary_str = sess.run(capsNet.train_summary)
                        sv.summary_writer.add_summary(summary_str, global_step)
                    # test
                    if (global_step + 1) % cfg.test_sum_freq == 0:
                        test_acc = 0
                        for i in range(num_test_batch):
                            start = i * cfg.batch_size
                            end = start + cfg.batch_size
                            test_acc += sess.run(capsNet.batch_accuracy, {capsNet.X: test_x[start:end],
                                                                          capsNet.labels: test_y[start:end]})
                        test_acc = test_acc / (cfg.batch_size * num_test_batch)
                        print("{} {}".format(epoch, test_acc))
                    pass
                # save model
                if epoch % cfg.save_freq == 0:
                    sv.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))
            pass

        pass

    pass

if __name__ == "__main__":
    Runner().run()
