import tensorflow as tf
import time
import math
import os
from dataset import Dataset
from model import SequenceModel

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('model_path', 'model/', 'the path of model')
tf.flags.DEFINE_string('data_path', 'data/origin', 'the path of data')


class Trainer:
    def __init__(self, data_dir, model_dir):
        self.graph = tf.Graph()
        self.model_dir = model_dir

        # Define train graph
        with self.graph.as_default():
            dataset = Dataset(data_dir)
            self.hparams = dataset.hparams
            self.model = SequenceModel(
                dataset.get_iterator(),
                self.hparams,
                mode='train'
            )

    def train(self):
        """ Train a seq2seq model. """

        # Write Summary
        writer = tf.summary.FileWriter(self.model_dir, graph=self.graph)
        num_epochs = self.hparams.num_epochs

        ckpt = tf.train.latest_checkpoint(self.model_dir)

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Restore latest checkpoint
            if ckpt:
                self.model.saver.restore(sess, ckpt)
                print('# load checkpoint {0} successfully'.format(ckpt))

            global_step = self.model.global_step.eval(session=sess)

            # Initialize all of the iterators
            sess.run(self.model.iterator.initializer)

            # Initialize the statistic variables
            ckpt_loss = 0.0
            step_start = global_step
            train_perp, last_record_perp = 2000.0, 2.0
            train_epoch = 0

            print("# Training loop started @ {}".format(
                time.strftime("%Y-%m-%d %H:%M:%S")))
            epoch_start_time = time.time()

            while train_epoch < num_epochs:
                learning_rate = self._get_learning_rate(train_perp)

                try:
                    # One step
                    step_start_time = time.time()
                    res = self.model.train(sess, learning_rate)
                    (_, step_loss, _, step_summary, global_step, _, _) = res
                    step_time = time.time() - step_start_time

                    print('epoch {:d}, step {:d}, loss is {:.3f}, {:.2f}s'.format(
                        train_epoch + 1, global_step, step_loss, step_time))
                    writer.add_summary(step_summary, global_step)
                    ckpt_loss += step_loss
                except tf.errors.OutOfRangeError:
                    train_epoch += 1

                    # Calculate perplexity
                    steps_count = global_step - step_start
                    mean_loss = ckpt_loss / steps_count
                    train_perp = math.exp(
                        float(mean_loss)) if mean_loss < 300 else math.inf
                    summary = tf.Summary(value=[tf.Summary.Value(
                        tag="train_perp", simple_value=train_perp)])
                    writer.add_summary(summary, global_step)

                    epoch_dur = time.time() - epoch_start_time
                    print("# Finished epoch {:2d} @ step {:5d} @ {}. In the epoch, learning rate = {:.6f}, "
                          "mean loss = {:.4f}, perplexity = {:8.4f}, and {:.2f} seconds elapsed."
                          .format(train_epoch, global_step, time.strftime("%Y-%m-%d %H:%M:%S"),
                                  learning_rate, mean_loss, train_perp, round(epoch_dur, 2)))

                    # Save checkpoint
                    if train_perp < 1.6 and train_perp < last_record_perp:
                        self.model.saver.save(
                            sess, os.path.join(self.model_dir, 'result'), global_step=global_step)
                    last_record_perp = train_perp

                    # Reset
                    epoch_start_time = time.time()  # The start time of the next epoch
                    ckpt_loss = 0.0
                    step_start = global_step
                    sess.run(self.model.iterator.initializer)

                    continue

            # Done training
            self.model.saver.save(sess, self.model_dir + 'result', global_step=global_step)
            writer.close()

    @staticmethod
    def _get_learning_rate(perplexity):
        if perplexity <= 1.48:
            return 9.6e-5
        elif perplexity <= 1.64:
            return 1e-4
        elif perplexity <= 2.0:
            return 1.2e-4
        elif perplexity <= 2.4:
            return 1.6e-4
        elif perplexity <= 3.2:
            return 2e-4
        elif perplexity <= 4.8:
            return 2.4e-4
        elif perplexity <= 8.0:
            return 3.2e-4
        elif perplexity <= 16.0:
            return 4e-4
        elif perplexity <= 32.0:
            return 6e-4
        else:
            return 8e-4


def main(_):
    trainer = Trainer(FLAGS.data_path, FLAGS.model_path)
    trainer.train()


if __name__ == "__main__":
    tf.app.run()