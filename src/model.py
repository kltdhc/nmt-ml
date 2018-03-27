import tensorflow as tf
from decoder import BasicDecoder
from tensorflow.python.layers import core as layers_core

class model():
    def __init__(self, num_input, w2v, maxsenlen, maxanslen, n_hidden=512, batch_size=32, learning_rate=0.1):
        self.insent = []
        self.inans = tf.placeholder(tf.int32, shape=[None, maxanslen], name='in_ans')
        self.inans_len = tf.placeholder(tf.int32, shape=[None], name='in_sent_len')
        self.batch_size = batch_size
        self.w2v = tf.Variable(w2v)
        self.maxsenlen = maxsenlen
        self.maxanslen = maxanslen
        self.output_layers = [layers_core.Dense(
            len(self.w2v), use_bias=False, name="output_projection") for _ in range(num_input)]
        for i in range(num_input):
            self.insent.append((
                tf.placeholder(tf.int32, shape=[None, maxsenlen], name='in_sent_%d'%i),
                tf.placeholder(tf.int32, shape=[None], name='in_sent_len_%d'%i)))
            # self.intar.append((
            #     tf.placeholder(tf.int32, shape=[None, maxsenlen], name='in_tar_%d'%i),
            #     tf.placeholder(tf.int32, shape=[None], name='in_tar_len_%d'%i)))
        
        emb_inputs = []
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            for i in range(len(self.insent)):
                emb_inputs.append(tf.nn.embedding_lookup(self.w2v, self.insent[i][0]))
        
        encoders = []
        for i in range(len(self.insent)):
            encoders.append(self.build_encoder(emb_inputs[i], self.insent[i][1], n_hidden))

        decoder = []
        self.cells = []
        self.steps = []
        for i in range(len(self.insent)):
            cell = self.build_decoder_cells(encoders[i], n_hidden)
            self.cells.append(cell)
            logits, sample_id, final_context_state = self.build_single_decoder(
                cell, encoders[i], self.inans, self.inans_len, self.output_layers[i], "decoder%d"%i)
            decoder.append(sample_id)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=logits, logits=self.inans)
            target_weights = tf.sequence_mask(
                self.inans_len, maxanslen, dtype=self.inans.dtype)
            train_loss = (tf.reduce_sum(crossent * target_weights) /
                self.batch_size)
            params = tf.trainable_variables()
            gradients = tf.gradients(train_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            update_step = optimizer.apply_gradients(
                zip(clipped_gradients, params))
            self.steps.append(update_step)
        
        # multilayer decoder
        self.test_logits, self.test_sample_id, final_context_state = self.build_multi_decoder(self.cells, encoders, self.output_layers, "ml_decoder")

    def build_encoder(self, input_s, inlen, n_hidden):
        # Build RNN cell
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

        # Run Dynamic RNN
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, input_s,
            sequence_length=inlen, time_major=True)
        return encoder_state

    def build_decoder_cells(self, input_state, n_hidden):
        # Build RNN cell
        # attention_states: [batch_size, max_time, num_units]
        attention_states = tf.transpose(input_state, [1, 0, 2])

        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            n_hidden, attention_states,
            memory_sequence_length=source_sequence_length)
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

        return tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mechanism,
            attention_layer_size=n_hidden)

    def build_single_decoder(self, cell, input_state, tar_in, tar_len, output_layer, scope):
        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            tar_in, tar_len, time_major=True)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell, helper, input_state, output_layer=output_layer)
        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            output_time_major=True,
            swap_memory=True,
            scope=scope)
        sample_id = outputs.sample_id
        # logits = self.output_layer(outputs.rnn_output)
        logits = outputs.rnn_output
        return logits, sample_id, final_context_state
    
    def build_multi_decoder(self, cells, input_states, output_layers, scope):
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.w2v, tf.fill([self.batch_size], 2), 3)
        my_decoder = BasicDecoder(
            cells,
            helper,
            input_states,
            output_layer=output_layers)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=True,
            swap_memory=True,
            maximum_iterations=self.maxanslen,
            scope=scope)
        logits = outputs.rnn_output
        sample_id = outputs.sample_id
        return logits, sample_id, final_context_state