import tensorflow as tf

class model():
    def __init__(self, num_input, w2v, maxsenlen, n_hidden):
        self.insent = []
        self.intar = []
        self.w2v = tf.Variable(w2v)
        for i in range(num_input):
            self.insent.append((
                tf.placeholder(tf.int32, shape=[None, maxsenlen], name='in_sent_%d'%i),
                tf.placeholder(tf.int32, shape=[None], name='in_sent_len_%d'%i)))
            self.intar.append((
                tf.placeholder(tf.int32, shape=[None, maxsenlen], name='in_tar_%d'%i),
                tf.placeholder(tf.int32, shape=[None], name='in_tar_len_%d'%i)))
        
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
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, 
            output_time_major=True,
            swap_memory=True,
            scope=scope)
        logits = outputs.rnn_output
    