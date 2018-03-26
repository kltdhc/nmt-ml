import tensorflow as tf

class model():
    def __init__(self, num_input):
        self.insent = []
        for i in range(num_input):
            self.insent.append((
                tf.placeholder(tf.int32, shape=[None, maxsenlen], name='in_sent_%d'%i),
                tf.placeholder(tf.int32, shape=[None], name='in_sent_len_%d'%i)))
        
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            for i in range(len(self.insent)):
                encoder_emb_inp = tf.nn.embedding_lookup(self.w2v, self.insent[i][0])
        # Build RNN cell
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        # Run Dynamic RNN
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, encoder_emb_inp,
            sequence_length=source_sequence_length, time_major=True)
        
