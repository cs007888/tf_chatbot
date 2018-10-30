import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
import numpy as np
import model_helper
import dataset
from word_util import WordUtil

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT


class SequenceModel:
    def __init__(self, iterator, hparams, mode, scope=None):
        self.iterator = iterator
        self.hparams = hparams
        self.mode = mode
        self.scope = scope

        # Initializer
        initializer = model_helper.get_initializer(
            self.hparams.init_op, None, self.hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        # Embeddings
        with tf.variable_scope(scope or 'embedding'):
            self.embedding = tf.get_variable(
                'embedding', [self.hparams.vocab_size, self.hparams.num_units], dtype=tf.float32)

        # Output Layer
        with tf.variable_scope(scope or "build_network"):
            with tf.variable_scope('decoder/output_projection'):
                self.output_layer = tf.layers.Dense(
                    self.hparams.vocab_size, use_bias=False)

        # Batch Size
        self.batch_size = tf.size(self.iterator.src_seq)

        # Build Graph
        print("# Building graph for the model ...")
        res = self.build_graph(self.scope)

        if self.mode == TRAIN:
            self.train_loss = res[1]
            self.word_count = tf.reduce_sum(
                tf.reduce_sum(self.iterator.src_seq) +
                tf.reduce_sum(self.iterator.tar_seq)
            )
        elif self.mode == EVAL:
            self.eval_loss = res[1]
        elif self.mode == PREDICT:
            self.infer_logits, _, self.final_state, self.sample_id = res

        if self.mode != PREDICT:
            # Count the number of predicted words for compute perplexity.
            self.predict_count = tf.reduce_sum(self.iterator.tar_seq)

        # Define variables
        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()

        # Optimizer
        if self.mode == TRAIN:
            self.learning_rate = tf.placeholder(
                tf.float32, shape=[], name='learning_rate')

            # self.learning_rate = tf.train.exponential_decay(
            #     0.001, self.global_step, 1000, 0.9)
            opt = tf.train.AdamOptimizer(self.learning_rate)

            # Gradient
            gradients = tf.gradients(
                self.train_loss,
                params,
                colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops
            )
            clipped_gradients, gradient_norm_summary, _ = model_helper.gradient_clip(
                gradients, self.hparams.max_gradient_norm)
            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), self.global_step)

            # Summary
            self.train_summary = tf.summary.merge([
                tf.summary.scalar('train_loss', self.train_loss),
                tf.summary.scalar('learning_rate', self.learning_rate)
            ] + gradient_norm_summary)
        else:
            self.infer_summary = tf.no_op()

        # Saver
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.hparams.max_to_keep)

    # Train Step
    def train(self, sess, learning_rate):
        assert self.mode == TRAIN
        return sess.run([
            self.update,
            self.train_loss,
            self.predict_count,
            self.train_summary,
            self.global_step,
            self.word_count,
            self.batch_size
        ], feed_dict={self.learning_rate: learning_rate})

    # Eval Step
    def eval(self, sess):
        assert self.mode == EVAL
        return sess.run([
            self.eval_loss,
            self.predict_count,
            self.batch_size
        ])

    # Inference Step
    def infer(self, sess):
        assert self.mode == PREDICT
        _, summary, sample_id = sess.run([
            self.infer_logits, self.infer_summary, self.sample_id
        ])

        if self.hparams.time_major:
            sample_id = sample_id.transpose()

        return sample_id, summary

    def build_graph(self, scope):
        print('# create {0} graph'.format(self.mode))
        dtype = tf.float32

        with tf.variable_scope(scope or 'dynamic_seq2seq', dtype=dtype):
            encoder_outputs, encoder_state = self._build_encoder()
            logits, sample_id, final_state = self._build_decoder(
                encoder_outputs, encoder_state)

            if self.mode != PREDICT:
                loss = self._compute_loss(logits)
            else:
                loss = None

            return logits, loss, final_state, sample_id

    # Encoder
    def _build_encoder(self):
        with tf.variable_scope('encoder'):
            src_input = self.iterator.src
            if self.hparams.time_major:
                inputs = tf.transpose(src_input)  # time major
            inputs = tf.nn.embedding_lookup(self.embedding, inputs)

            if self.hparams.encoder_type == 'bi':
                outputs, state = self._build_bidirectional_rnn(
                    inputs, self.iterator.src_seq)
            else:
                cell = self._build_encoder_cell(self.hparams.num_layers)
                outputs, state = tf.nn.dynamic_rnn(
                    cell, inputs, self.iterator.src_seq, dtype=tf.float32, time_major=self.hparams.time_major)

        return outputs, state

    def _build_encoder_cell(self, num_layers):
        return model_helper.build_rnn_cell(
            self.hparams.unit_type,
            self.hparams.num_units,
            num_layers,
            self.hparams.dropout,
        )

    def _build_bidirectional_rnn(self, inputs, sequence_length):
        ''' Build bidirectional rnn '''
        num_bi_layers = int(self.hparams.num_layers / 2)
        fw_cell = self._build_encoder_cell(num_bi_layers)
        bw_cell = self._build_encoder_cell(num_bi_layers)
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            sequence_length,
            dtype=tf.float32,
            time_major=self.hparams.time_major,
            swap_memory=True
        )
        state = []
        for i in range(num_bi_layers):
            state.append(bi_state[0][i])
            state.append(bi_state[1][i])
        state = tuple(state)

        return tf.concat(bi_outputs, -1), state

    # Decoder
    def _build_decoder(self, encoder_outputs, encoder_state):
        sos_id = WordUtil.START
        eos_id = WordUtil.END

        # maximum_iteration: The maximum decoding steps.
        if self.hparams.tgt_max_len_infer:
            maximum_iterations = self.hparams.tgt_max_len_infer
        else:
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(self.iterator.src_seq)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))

        with tf.variable_scope('decoder') as decoder_scope:
            cell, initial_state = self._build_decoder_cell(
                encoder_outputs, encoder_state, self.iterator.src_seq)
            if self.mode != PREDICT:
                tar_inputs = self.iterator.tar_in
                if self.hparams.time_major:
                    tar_inputs = tf.transpose(tar_inputs)  # time major
                tar_inputs = tf.nn.embedding_lookup(
                    self.embedding, tar_inputs)

                helper = seq2seq.TrainingHelper(
                    tar_inputs, self.iterator.tar_seq, time_major=self.hparams.time_major)
                decoder = seq2seq.BasicDecoder(
                    cell, helper, initial_state)
                outputs, state, _ = seq2seq.dynamic_decode(
                    decoder,
                    output_time_major=True,
                    swap_memory=True,
                    scope=decoder_scope
                )
                logits = self.output_layer(outputs.rnn_output)
                sample_id = outputs.sample_id
            else:
                beam_width = self.hparams.beam_width
                length_penalty_weight = self.hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], sos_id)
                end_token = eos_id
                # BeamSearch mode
                if beam_width > 0:
                    decoder = seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=initial_state,
                        beam_width=beam_width,
                        output_layer=self.output_layer,
                        length_penalty_weight=length_penalty_weight
                    )
                # GreedySearch mode
                else:
                    helper = seq2seq.GreedyEmbeddingHelper(
                        embedding=self.embedding,
                        start_tokens=start_tokens,
                        end_token=end_token
                    )
                    decoder = seq2seq.BasicDecoder(
                        cell, helper, initial_state, self.output_layer)

                outputs, state, _ = seq2seq.dynamic_decode(
                    decoder,
                    output_time_major=self.hparams.time_major,
                    maximum_iterations=maximum_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                )
                if beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, state

    def _build_decoder_cell(self, encoder_outputs, encoder_state,
                            source_sequence_length):
        beam_width = self.hparams.beam_width
        if self.hparams.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])

        if self.mode == PREDICT and beam_width > 0:
            memory = seq2seq.tile_batch(memory, beam_width)
            source_sequence_length = seq2seq.tile_batch(
                source_sequence_length, beam_width)
            encoder_state = seq2seq.tile_batch(encoder_state, beam_width)
            batch_size = self.batch_size * beam_width
        else:
            batch_size = self.batch_size

        # Use Attention Mechanism
        attention_machanism = seq2seq.LuongAttention(
            num_units=self.hparams.num_units,
            memory=memory,
            memory_sequence_length=source_sequence_length
        )

        cell = model_helper.build_rnn_cell(
            self.hparams.unit_type,
            self.hparams.num_units,
            self.hparams.num_layers,
            self.hparams.dropout,
        )
        alignment_history = (
            self.mode == PREDICT and beam_width == 0)

        cell = seq2seq.AttentionWrapper(
            cell,
            attention_machanism,
            attention_layer_size=self.hparams.num_units,
            alignment_history=alignment_history,
            name='attention'
        )
        if self.hparams.pass_hidden_state:
            initial_state = cell.zero_state(
                batch_size, tf.float32).clone(cell_state=encoder_state)
        else:
            initial_state = cell.zero_state(batch_size, tf.float32)

        return cell, initial_state

    def _compute_loss(self, logits):
        target_output = self.iterator.tar_out
        if self.hparams.time_major:
            logits = tf.transpose(logits, [1, 0, 2])
        mask = tf.sequence_mask(
            self.iterator.tar_seq, target_output.shape[1].value, logits.dtype)
        # loss = tf.losses.sparse_softmax_cross_entropy(
        #     self.iterator.tar_out, logits, weights=mask)
        loss = seq2seq.sequence_loss(
            logits, target_output, weights=mask)
        return loss
