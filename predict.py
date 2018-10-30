import tensorflow as tf
from model import SequenceModel
from dataset import Dataset
from word_util import WordUtil

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('model_path', 'model/', 'the path of model')
tf.flags.DEFINE_string('data_path', 'data/origin', 'the path of vocab')


class Predictor():
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.graph = tf.Graph()

    def predict(self, sentence):
        with self.graph.as_default():  
            dataset = Dataset(self.data_path)
            batch_input = dataset.get_infer_iterator(sentence)
            model = SequenceModel(batch_input, dataset.hparams, mode='infer')
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            ckpt = tf.train.latest_checkpoint(self.model_path)
            saver.restore(sess, ckpt)
            res = model.infer(sess)
            arr = []
            if dataset.hparams.beam_width == 0:
                for i in res[0]:
                    _s = []
                    for s in i:
                        _s.append(s)
                    arr.append(_s)
            else:
                for i in res[0]:
                    _s = []
                    for s in i[0]:
                        _s.append(s)
                    arr.append(_s)
        return arr


def main(_):
    predictor = Predictor(FLAGS.data_path, FLAGS.model_path)

    s = input('输入前句：')
    util = WordUtil()
    arr = predictor.predict(s)
    for index in arr:
        reply = util.reverse_transform(index)
        print(reply)


if __name__ == '__main__':
    tf.app.run()