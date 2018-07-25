import tensorflow as tf
from decoder2 import BasicDecoder
from tensorflow.python.layers import core as layers_core
import numpy as np
from bleu import compute_bleu
from copynet import CopyNetWrapper

def eval(tar, ref):
    tar = [x[:(x+[3]).index(3)] for x in tar]
    ref = [x[:(x+[3]).index(3)] for x in ref]
    return compute_bleu(tar, ref)[0]

class model():
    def __init__(self, num_input, w2v, maxsenlen, maxanslen, n_hidden=512, batch_size=32, 
        learning_rate=1., max_gradient_norm=5., layers=None, dropout=.2, birnn=None, copynet=False, adam=False):
        self.insent = []
        self.num_input = num_input
        self.batch_size = batch_size
        self.inans = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='in_ans')
        self.inans_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='in_sent_len')
        self.inans_train = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='in_ans_tar_in')
        self.w2v = []
        self.vocab_shape = len(w2v)
        self.copynet = copynet
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        if birnn is None:
            birnn = [True for _ in range(num_input)]
        if layers is None:
            layers = [2 for _ in range(num_input)]
        for i in range(num_input):
            self.w2v.append(tf.concat(
                [tf.constant([[0. for _ in range(len(w2v[0]))]]), tf.Variable(w2v[1:], dtype=tf.float32, trainable=True)], 
                axis=0, name='w2v_l%d'%i))
        
        self.maxsenlen = maxsenlen
        self.maxanslen = maxanslen
        # self.output_layers = [layers_core.Dense(
        #     len(w2v), use_bias=False, name="output_projection") for _ in range(num_input)]
        self.output_layers = None
        for i in range(num_input):
            self.insent.append((
                tf.placeholder(tf.int32, shape=[self.batch_size, maxsenlen], name='in_sent_l%d'%i),
                tf.placeholder(tf.int32, shape=[self.batch_size], name='in_sent_len_l%d'%i)))
            # self.intar.append((
            #     tf.placeholder(tf.int32, shape=[None, maxsenlen], name='in_tar_l%d'%i),
            #     tf.placeholder(tf.int32, shape=[None], name='in_tar_len_l%d'%i)))
        
        emb_inputs = []
        self.tar_inputs = []
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            for i in range(num_input):
                self.tar_inputs.append(tf.nn.embedding_lookup(self.w2v[i], self.inans_train, name='tar_in_l%d'%i))
                emb_inputs.append(tf.nn.embedding_lookup(self.w2v[i], self.insent[i][0], name='encoder_in_l%d'%i))
        
        self.encoders = []
        self.encoders_out = []
        for i in range(num_input):
            encoder_out, encoder_state = self.build_encoder(emb_inputs[i], self.insent[i][1], n_hidden, i, layers[i], dropout, birnn[i])
            self.encoders_out.append(encoder_out)
            self.encoders.append(encoder_state)

        self.cells = []
        self.origin_cells = []
        self.losses = []
        self.train_outputs = []
        self.test_outputs = []
        for i in range(num_input):
            cell = self.build_decoder_cells(self.encoders_out[i], self.insent[i][1], n_hidden, i, layers[i])
            self.origin_cells.append(cell)
            if copynet:
                with tf.variable_scope('copynet_cell_l%d'%i):
                    cell = CopyNetWrapper(cell, self.encoders_out[i], self.insent[i][0], len(w2v), len(w2v))
            self.cells.append(cell)
            if self.output_layers is None:
                opl = None
            else:
                opl = self.output_layers[i]
            logits, sample_id, final_context_state = self.build_single_decoder(
                cell, self.encoders[i], self.tar_inputs[i], self.inans_len, opl, "decoder_l%d"%i, i)
            self.train_outputs.append(sample_id)
            logits2, sample_id2, final_context_state2 = self.build_single_test_decoder(
                cell, self.encoders[i], opl, "test_decoder_l%d"%i, i)
            self.test_outputs.append(sample_id2)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.inans)
            target_weights = tf.sequence_mask(
                self.inans_len, tf.shape(logits)[1], dtype=logits.dtype)
            train_loss = (tf.reduce_sum(crossent * target_weights) /
                self.batch_size)
            self.losses.append(train_loss)
            
        params = tf.trainable_variables()
        self.steps = []
        for i in range(num_input):
            now_params = []
            for j in params:
                if '_l%d'%i in j.name:
                    now_params.append(j)
            gradients = tf.gradients(self.losses[i], now_params)
            # print(gradients)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, max_gradient_norm)
            if adam:
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            update_step = optimizer.apply_gradients(
                zip(clipped_gradients, now_params))
            self.steps.append(update_step)
        
        # self.build_ml_decoder()
        
    def build_ml_decoder(self, models=None, prop=None, name=''):
        # multilayer decoder
        if models is None:
            cells = self.cells
            encoders = self.encoders
            ol = self.output_layers
            ti = self.tar_inputs
        else:
            cells = [self.cells[i] for i in models]
            encoders = [self.encoders[i] for i in models]
            if self.output_layers is not None:
                ol = [self.output_layers[i] for i in models]
            else:
                ol = None
            ti = [self.tar_inputs[i] for i in models]
            models = list(range(self.num_input))

        with tf.variable_scope("ml_unused_prop_"+name):
            prop = [1. for _ in range(len(cells))]
            prop = tf.Variable(prop)
            # prop = None
            # dense_layer = tf.layers.Dense(
            #     1, 
            #     dtype=self.w2v[0].dtype.base_dtype)
            dense_layer = None

        self.test_logits, self.test_sample_id, final_context_state = \
            self.build_multi_decoder(cells, encoders, ol, "ml_decoder", prop, dense_layer)
        tlogits, t_sid, final_context_state = \
            self.build_multi_decoder_for_train(cells, encoders, ti, self.inans_len, ol, "ml_train", prop, dense_layer)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tlogits, labels=self.inans)
        target_weights = tf.sequence_mask(
            self.inans_len, tf.shape(tlogits)[1], dtype=tlogits.dtype)
        self.ml_loss = (tf.reduce_sum(crossent * target_weights) /
            self.batch_size)
        
        params = tf.trainable_variables()
        now_params = []
        for j in params:
            if 'ml_prop' in j.name:
                now_params.append(j)
        if len(now_params) > 0:
            gradients = tf.gradients(self.ml_loss, now_params)
            # print(gradients)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.ml_update_step = optimizer.apply_gradients(
                zip(clipped_gradients, now_params))

    def save_model(self, sess, path, step, keyword=''):
        all_vars = tf.trainable_variables()
        needed = []
        for i in all_vars:
            if keyword in i.name:
                needed.append(i)
        saver = tf.train.Saver(needed)
        saver.save(sess, path, global_step=step) 
    
    def restore_model(self, sess, path, step, keyword=''):
        all_vars = tf.trainable_variables()
        needed = []
        for i in all_vars:
            if keyword in i.name:
                needed.append(i)
        saver = tf.train.Saver(needed)
        saver.restore(sess, path+'-%d'%step) 

    def build_encoder(self, input_s, inlen, n_hidden, i, layers, dropout=0.2, birnn=True):
        with tf.variable_scope('build_encoder_l%d'%i):
            # Build RNN cell
            if layers>1:
                f_encoder_cell = [tf.contrib.rnn.DropoutWrapper(
                    cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden), input_keep_prob=(1.0 - dropout))]
                for i in range(layers-1):
                    f_encoder_cell.append(tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.DropoutWrapper(
                        cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden), input_keep_prob=(1.0 - dropout))))
                f_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(f_encoder_cell)
                if birnn:
                    b_encoder_cell = [tf.contrib.rnn.DropoutWrapper(
                        cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden), input_keep_prob=(1.0 - dropout))]
                    for i in range(layers-1):
                        b_encoder_cell.append(tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.DropoutWrapper(
                            cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden), input_keep_prob=(1.0 - dropout))))
                    b_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(b_encoder_cell)
            else:
                f_encoder_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden), input_keep_prob=(1.0 - dropout))
                if birnn:
                    b_encoder_cell = tf.contrib.rnn.DropoutWrapper(
                        cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden), input_keep_prob=(1.0 - dropout))
            # Run Dynamic RNN
            #   encoder_outputs: [max_time, batch_size, num_units]
            #   encoder_state: [batch_size, num_units]
            if birnn:
                encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                    f_encoder_cell, b_encoder_cell, input_s,
                    sequence_length=inlen, dtype=tf.float32)
                return tf.add_n(encoder_outputs), encoder_state[1]
            else:
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    f_encoder_cell, input_s,
                    sequence_length=inlen, dtype=tf.float32)
                return encoder_outputs, encoder_state

    def build_decoder_cells(self, encoder_out, inlen, n_hidden, i, layers, dropout=.2):
        # Build RNN cell
        # attention_states: [batch_size, max_time, num_units]
        with tf.variable_scope('build_decoder_l%d'%i):
            # attention_states = tf.transpose(input_state, [1, 0, 2])

            # Create an attention mechanism
            if layers>1:
                cell = [tf.contrib.rnn.DropoutWrapper(
                    cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden), input_keep_prob=(1.0 - dropout))]
                for i in range(layers-1):
                    cell.append(tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.DropoutWrapper(
                        cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden), input_keep_prob=(1.0 - dropout))))
                cell = tf.nn.rnn_cell.MultiRNNCell(cell)
            else:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden), input_keep_prob=(1.0 - dropout))

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                n_hidden, 
                encoder_out,
                memory_sequence_length=inlen, 
                scale=True)

            return tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism,
                attention_layer_size=n_hidden)

    def build_single_decoder(self, cell, input_state, tar_in, tar_len, output_layer, scope, i):
        # Helper
        with tf.variable_scope(scope):
            helper = tf.contrib.seq2seq.TrainingHelper(
                tar_in, tar_len)
            # Decoder
            if self.copynet:
                input_state = self.origin_cells[i].zero_state(self.batch_size, tf.float32).clone(
                    cell_state=input_state)
            decoder_initial_state = cell.zero_state(self.batch_size, tf.float32).clone(
                cell_state=input_state)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell, helper, decoder_initial_state, output_layer=output_layer)
            # Dynamic decoding
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                # output_time_major=True,
                swap_memory=True,
                maximum_iterations=self.maxanslen,
                scope=scope)
            sample_id = outputs.sample_id
            # logits = self.output_layer(outputs.rnn_output)
            logits = outputs.rnn_output
        return logits, sample_id, final_context_state

    def build_single_test_decoder(self, cell, input_state, output_layer, scope, i):
        with tf.variable_scope(scope):
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.w2v[i], tf.fill([self.batch_size], 2), 3)
            # Decoder
            if self.copynet:
                input_state = self.origin_cells[i].zero_state(self.batch_size, tf.float32).clone(
                    cell_state=input_state)
            decoder_initial_state = cell.zero_state(self.batch_size, tf.float32).clone(
                cell_state=input_state)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell, helper, decoder_initial_state, output_layer=output_layer)
            # Dynamic decoding
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                # output_time_major=True,
                swap_memory=True,
                maximum_iterations=self.maxanslen,
                scope=scope)
            sample_id = outputs.sample_id
            # logits = self.output_layer(outputs.rnn_output)
            logits = outputs.rnn_output
        return logits, sample_id, final_context_state
    
    def build_multi_decoder_for_train(self, cells, input_states, tar_in, tar_len, output_layers, scope, prop=None, dense_layer=None):
        helpers = []
        for i in range(len(cells)):
            helpers.append(tf.contrib.seq2seq.TrainingHelper(tar_in[i], tar_len))

        if self.copynet:
            input_states = [self.origin_cells[i].zero_state(self.batch_size, tf.float32).clone(
                cell_state=input_states[i]) for i in range(len(cells))]
        decoder_initial_states = [cell.zero_state(self.batch_size, tf.float32).clone(
            cell_state=input_state) for cell, input_state in zip(cells, input_states)]
        my_decoder = BasicDecoder(
            cells,
            helpers,
            decoder_initial_states,
            output_layers=output_layers,
            prop = prop, 
            dense_layer = dense_layer)
        
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            # output_time_major=True,
            swap_memory=True,
            maximum_iterations=self.maxanslen,
            scope = scope)
        logits = outputs.rnn_output
        sample_id = outputs.sample_id
        return logits, sample_id, final_context_state

    def build_multi_decoder(self, cells, input_states, output_layers, scope, prop=None, dense_layer=None):
        helpers = []
        for i in range(len(cells)):
            helpers.append(tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.w2v[i], tf.fill([self.batch_size], 2), 3))
            helpers.append(tf.contrib.seq2seq.)
        
        if self.copynet:
            input_states = [self.origin_cells[i].zero_state(self.batch_size, tf.float32).clone(
                cell_state=input_states[i]) for i in range(len(cells))]
        decoder_initial_states = [cell.zero_state(self.batch_size, tf.float32).clone(
            cell_state=input_state) for cell, input_state in zip(cells, input_states)]
        my_decoder = BasicDecoder(
            cells,
            helpers,
            decoder_initial_states,
            output_layers=output_layers, 
            prop=prop,
            dense_layer = dense_layer)
        
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            # output_time_major=True,
            swap_memory=True,
            maximum_iterations=self.maxanslen,
            scope=scope)
        logits = outputs.rnn_output
        sample_id = outputs.sample_id
        return logits, sample_id, final_context_state

    def train_all(self, sess, in_sens, in_sens_len, in_ans, in_ans_len, states=None):
        for i in range(len(in_sens)):
            if states is None or states[i]:
                loss = self.train(sess, i, in_sens[i], in_sens_len[i], in_ans, in_ans_len)
                rt = self.test_sep(sess, i, in_sens[i], in_sens_len[i])
                print('avg_loss = %f, train...%d...bleu='%(loss, i), eval(in_ans, rt))

    def get_train_batch(self, batch_no, num_model, in_sens, in_sens_len, in_ans=None, in_ans_len=None):
        num_zero = self.batch_size - len(in_sens[batch_no*self.batch_size:(1+batch_no)*self.batch_size])
        empty_s = [0 for i in range(len(in_sens[0]))]
        feed_dict = {}
        feed_dict[self.insent[num_model][0]] = \
            in_sens[batch_no*self.batch_size:(1+batch_no)*self.batch_size] + [empty_s for i in range(num_zero)]
        feed_dict[self.insent[num_model][1]] = \
            in_sens_len[batch_no*self.batch_size:(1+batch_no)*self.batch_size] + [1 for i in range(num_zero)]
        if in_ans is not None:
            empty_a = [0 for i in range(len(in_ans[0]))]
            amaxlen = max(in_ans_len[batch_no*self.batch_size:(1+batch_no)*self.batch_size])
            x = in_ans[batch_no*self.batch_size:(1+batch_no)*self.batch_size] + [empty_s for i in range(num_zero)]
            feed_dict[self.inans] = [i[:amaxlen] for i in x]
            feed_dict[self.inans_train] = [[2]+i[:amaxlen-1] for i in x]
            feed_dict[self.inans_len] = \
                in_ans_len[batch_no*self.batch_size:(1+batch_no)*self.batch_size] + [1 for i in range(num_zero)]
        return feed_dict
    
    def get_train_batch_ml(self, batch_no, in_sens, in_sens_len, in_ans=None, in_ans_len=None):
        num_zero = self.batch_size - len(in_sens[0][batch_no*self.batch_size:(1+batch_no)*self.batch_size])
        empty_s = [0 for i in range(len(in_sens[0][0]))]
        feed_dict = {}
        for num_model in range(len(in_sens)):
            feed_dict[self.insent[num_model][0]] = \
                in_sens[num_model][batch_no*self.batch_size:(1+batch_no)*self.batch_size] + [empty_s for i in range(num_zero)]
            feed_dict[self.insent[num_model][1]] = \
                in_sens_len[num_model][batch_no*self.batch_size:(1+batch_no)*self.batch_size] + [1 for i in range(num_zero)]
        if in_ans is not None:
            empty_a = [0 for i in range(len(in_ans[0]))]
            amaxlen = max(in_ans_len[batch_no*self.batch_size:(1+batch_no)*self.batch_size])
            x = in_ans[batch_no*self.batch_size:(1+batch_no)*self.batch_size] + [empty_s for i in range(num_zero)]
            feed_dict[self.inans] = [i[:amaxlen] for i in x]
            feed_dict[self.inans_train] = [[2]+i[:amaxlen-1] for i in x]
            feed_dict[self.inans_len] = \
                in_ans_len[batch_no*self.batch_size:(1+batch_no)*self.batch_size] + [1 for i in range(num_zero)]
        return feed_dict

    def get_test_batch(self, batch_no, in_sens, in_sens_len):
        num_zero = self.batch_size - len(in_sens[0][batch_no*self.batch_size:(1+batch_no)*self.batch_size])
        empty_s = [0 for i in range(len(in_sens[0][0]))]
        feed_dict = {}
        for num_model in range(len(in_sens)):
            feed_dict[self.insent[num_model][0]] = \
                in_sens[num_model][batch_no*self.batch_size:(1+batch_no)*self.batch_size] + [empty_s for i in range(num_zero)]
            feed_dict[self.insent[num_model][1]] = \
                in_sens_len[num_model][batch_no*self.batch_size:(1+batch_no)*self.batch_size] + [1 for i in range(num_zero)]
        return feed_dict

    def train(self, sess, num_model, in_sens, in_sens_len, in_ans, in_ans_len):
        batch_num = len(in_sens) // self.batch_size
        all_loss = 0
        shuffle_indices = np.random.permutation(np.arange(len(in_sens)))
        in_sens = np.array(in_sens)[shuffle_indices].tolist()
        in_sens_len = np.array(in_sens_len)[shuffle_indices].tolist()
        in_ans = np.array(in_ans)[shuffle_indices].tolist()
        in_ans_len = np.array(in_ans_len)[shuffle_indices].tolist()
        for i in range(batch_num):
            feed_dict = self.get_train_batch(i, num_model, in_sens, in_sens_len, in_ans, in_ans_len)
            loss, _ = sess.run([self.losses[num_model], self.steps[num_model]], feed_dict=feed_dict)
            all_loss += loss
        return all_loss / batch_num
    
    def train_ml(self, sess, in_sens, in_sens_len, in_ans, in_ans_len):
        batch_num = len(in_sens[0]) // self.batch_size
        all_loss = 0
        shuffle_indices = np.random.permutation(np.arange(len(in_sens[0])))
        for i in range(len(in_sens)):
            in_sens[i] = np.array(in_sens[i])[shuffle_indices].tolist()
            in_sens_len[i] = np.array(in_sens_len[i])[shuffle_indices].tolist()
        in_ans = np.array(in_ans)[shuffle_indices].tolist()
        in_ans_len = np.array(in_ans_len)[shuffle_indices].tolist()
        for i in range(min(batch_num, 50)):
            feed_dict = self.get_train_batch_ml(i, in_sens, in_sens_len, in_ans, in_ans_len)
            loss, _ = sess.run([self.ml_loss, self.ml_update_step], feed_dict=feed_dict)
            all_loss += loss
        return all_loss / batch_num

    def test(self, sess, in_sens, in_sens_len):
        batch_num = (len(in_sens[0])-1) // self.batch_size + 1
        all_ans = None
        all_logits = None
        for i in range(batch_num):
            feed_dict = self.get_test_batch(i, in_sens, in_sens_len)
            logits, ids = sess.run([self.test_logits, self.test_sample_id], feed_dict=feed_dict)
            if all_ans is None:
                all_ans = ids.tolist()
                all_logits = logits.tolist()
            else:
                all_ans = all_ans + ids.tolist()
                all_logits = all_logits + logits.tolist()
        return all_ans[:len(in_sens[0])], all_logits[:len(in_sens[0])]

    def test_sep(self, sess, num_model, in_sens, in_sens_len):
        batch_num = len(in_sens) // self.batch_size + 1
        all_loss = 0
        all_ans = []
        for i in range(batch_num):
            feed_dict = self.get_train_batch(i, num_model, in_sens, in_sens_len)
            output = sess.run(self.test_outputs[num_model], feed_dict=feed_dict)
            all_ans += output.tolist()
        return all_ans