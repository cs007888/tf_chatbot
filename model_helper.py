import tensorflow as tf
import numpy as np


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(
            seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(
            seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


def build_rnn_cell(cell_type, n_hidden, num_layres, dropout):
    return tf.nn.rnn_cell.MultiRNNCell(
        [_single_cell(cell_type, n_hidden, dropout) for _ in range(num_layres)])


def _single_cell(cell_type, n_hidden, dropout):
    if cell_type == 'gru':
        rnn = tf.nn.rnn_cell.GRUCell
    else:
        rnn = tf.nn.rnn_cell.BasicLSTMCell
    cell = rnn(n_hidden)
    if dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, output_keep_prob=1-dropout)
    return cell


def gradient_clip(gradients, max_gradient_norm):
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar('gradient_norm', gradient_norm)]
    gradient_norm_summary.append(
        tf.summary.scalar('clipped_gradient',
                          tf.global_norm(clipped_gradients))
    )
    return clipped_gradients, gradient_norm_summary, gradient_norm
