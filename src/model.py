import tensorflow as tf
from decoder import BasicDecoder

class model():
    def __init__(self, num_input, w2v, maxsenlen, maxanslen, n_hidden, batch_size):
        self.insent = []
        self.inans = tf.placeholder(tf.int32, shape=[None, maxanslen], name='in_ans')
        self.inans_len = tf.placeholder(tf.int32, shape=[None], name='in_sent_len')
        self.batch_size = batch_size
        self.w2v = tf.Variable(w2v)
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

        decoder_cells = []
        for i in range(len(self.insent)):
            decoder_cells.append(self.build_decoder_cells(n_hidden))
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=decoder_cells[-1][0], logits=self.inans)
            target_weights = tf.sequence_mask(
                self.inans_len, maxanslen, dtype=self.inans.dtype)
            train_loss = (tf.reduce_sum(crossent * target_weights) /
                self.batch_size)

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

    def build_decoder_cells(self, n_hidden):
        # Build RNN cell
        return tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    def build_single_decoder(self, cell, input_state, tar_in, tar_len, scope):
        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            tar_in, tar_len, time_major=True)
        # Decoder
        decoder = BasicDecoder(
            cell, helper, input_state)
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
    
    def build_multi_decoder(self, cells, input_states, scope):
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.w2v, tf.fill([self.batch_size], 2), 3)
        my_decoder = BasicDecoder(
            cells,
            helper,
            input_states,)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=True,
            swap_memory=True,
            scope=scope)
        logits = outputs.rnn_output
        sample_id = outputs.sample_id
        return logits, sample_id, final_context_state