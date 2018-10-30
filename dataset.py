import tensorflow as tf
import os
import collections
import json
import pandas as pd
import numpy as np
from word_util import WordUtil


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "src", "tar_in",
                            "tar_out", "src_seq",
                            "tar_seq"))):
    pass


class Dataset:
    def __init__(self, data_dir):
        # Find source path
        question_paths = tf.gfile.Glob(os.path.join(data_dir, '*question*.txt'))
        assert len(question_paths) > 0
        answer_paths = tf.gfile.Glob(os.path.join(data_dir, '*answer*.txt'))
        assert len(answer_paths) > 0
        hparams_paths = tf.gfile.Glob(os.path.join(data_dir, '*.json'))
        assert len(hparams_paths) > 0

        # Generate vocab
        self.util = WordUtil(question_paths[0], answer_paths[0])
        self.util.generate_vocab()
        self.util.generate_csv()
        self.csv_path = self.util.csv_file_path

        vocab_size = len(self.util.dict)
        self.hparams = self.get_hparams(hparams_paths[0], vocab_size)

    def get_iterator(self):
        csv = pd.read_csv(self.csv_path)

        sos_id = self.util.START
        eos_id = self.util.END

        src_dataset = tf.data.Dataset.from_tensor_slices(csv['question'])
        tar_dataset = tf.data.Dataset.from_tensor_slices(csv['answer'])

        dataset = tf.data.Dataset.zip((src_dataset, tar_dataset))
        dataset = dataset.shuffle(5000)
        dataset = dataset.map(
            lambda src, tar: (tf.string_split(
                [src], ' ').values, tf.string_split([tar], ' ').values)
        ).map(
            lambda src, tar: (tf.string_to_number(src, tf.int32),
                              tf.string_to_number(tar, tf.int32))
        ).map(
            lambda src, tar: (src, tf.concat(
                ([sos_id], tar), 0), tf.concat((tar, [eos_id]), 0))
        ).map(
            lambda src, tar_in, tar_out: (
                src, tar_in, tar_out, tf.size(src), tf.size(tar_in))
        ).padded_batch(
            self.hparams.batch_size,
            padded_shapes=([None], [None], [None], [], []),
            padding_values=(eos_id, eos_id, eos_id, 0, 0)
        ).prefetch(self.hparams.batch_size * 2)
        iterator = dataset.make_initializable_iterator()
        initializer = iterator.initializer
        result = iterator.get_next()
        return BatchedInput(
            initializer=initializer,
            src=result[0],
            tar_in=result[1],
            tar_out=result[2],
            src_seq=result[3],
            tar_seq=result[4])

    def get_infer_iterator(self, sentence):
        index = self.util.transform(sentence)
        print(index)

        seq = np.array([len(index)])
        index = np.array(index)[np.newaxis, :]
        return BatchedInput(
            initializer=None,
            src=index,
            tar_in=None,
            tar_out=None,
            src_seq=seq,
            tar_seq=None
        )

    def get_hparams(self, path, vocab_size):
        ''' Load hparams from json file '''
        with open(path, 'r') as f:
            values = json.load(f)
        hparams = tf.contrib.training.HParams(**values)
        hparams.add_hparam('vocab_size', vocab_size)
        return hparams
