import numpy as np
from model2 import model as Model
import tensorflow as tf
from bleu import compute_bleu
import time
import re

def build_vocab(sents, vocab=None, vocab_dict=None, add_word=True, fix_maxlen=None):
    if vocab is None and vocab_dict is not None:
        return
    if vocab is None:
        vocab = ['<space>', '<unk>', '<s>', '</s>']
    if vocab_dict is None:
        vocab_dict = {}
        count = 0
        for i in vocab:
            vocab_dict[i] = count
            count += 1
    cut_sents = []
    maxlen = 0
    for sent in sents:
        cut_sent = sent
        # cut_sent_num = [vocab_dict['<s>']]
        cut_sent_num = []
        for i in cut_sent:
            if i == '' :
                continue
            if vocab_dict.get(i) == None:
                if add_word:
                    vocab_dict[i] = len(vocab_dict)
                    vocab.append(i)
                else:
                    i = '<unk>'
            cut_sent_num.append(vocab_dict[i])
        if len(cut_sent_num) > maxlen:
            maxlen = len(cut_sent_num)
        cut_sent_num.append(vocab_dict['</s>'])
        cut_sents.append(cut_sent_num)

    if fix_maxlen is None:
        maxlen = maxlen + 10
    else:
        maxlen = fix_maxlen

    cut_len = []
    for i in range(len(cut_sents)):
        cut_len.append(len(cut_sents[i]))
        if len(cut_sents[i])<maxlen:
            cut_sents[i] += [vocab_dict['</s>'] for _ in range(maxlen - len(cut_sents[i]))]
        if len(cut_sents[i])>maxlen:
            cut_len[-1] = maxlen
            cut_sents[i] = cut_sents[i][:maxlen]
    
    return cut_sents, cut_len, vocab, vocab_dict

def build_vocab_matrix(vocab, vocab_dict, dir='../data/word_emb'):
    infile = open('../data/word_emb', encoding='utf8')
    embeddings = {}
    emb_size = 0
    for line in infile:
        line = line.strip().split(' ')
        if emb_size == 0 or emb_size == len(line)-1:
            if vocab_dict.get(line[0]) != None:
                embeddings[line[0]] = [float(x) for x in line[1:]]
                emb_size = len(line)-1
    embeddings['<space>'] = [0. for _ in range(emb_size)]
    return [embeddings.get(word, list(np.random.normal(loc=0., scale=np.sqrt(1./emb_size), size=emb_size))) for word in vocab]

def read_input(files):
    rt = []
    maxlen = 0
    for i in files:
        nowfile = open(i, encoding='utf8')
        nowrt = []
        for line in nowfile:
            line = line.strip().split(' ')
            nowrt.append(line)
            if len(line)>maxlen:
                maxlen = len(line)
        rt.append(nowrt)
    return rt, maxlen

def read_trainset(dirs):
    train_q, qmaxlen = read_input(['../data/'+i+'/train.q' for i in dirs])
    qmaxlen += 10
    train_a, amaxlen = read_input(['../data/'+i+'/train.a' for i in dirs])
    amaxlen += 10
    train_qsent = []
    train_qlen = []
    vocab = None
    vocab_dict = None
    for i in train_q:
        cut_sents, cut_len, vocab, vocab_dict = build_vocab(i, vocab, vocab_dict, fix_maxlen=qmaxlen)
        train_qsent.append(cut_sents)
        train_qlen.append(cut_len)
    train_asent = []
    train_alen = []
    for i in train_a:
        cut_sents, cut_len, vocab, vocab_dict = build_vocab(i, vocab, vocab_dict, fix_maxlen=amaxlen)
        train_asent.append(cut_sents)
        train_alen.append(cut_len)
    return train_qsent, train_qlen, train_asent, train_alen, vocab, vocab_dict, qmaxlen, amaxlen

def read_dev_test_set(name, dirs, vocab, vocab_dict, qmaxlen, amaxlen, read_ans=True):
    test_q, _ = read_input(['../data/'+i+'/'+name+'.q' for i in dirs])
    test_qsent = []
    test_qlen = []
    for i in test_q:
        cut_sents, cut_len, vocab, vocab_dict = build_vocab(i, vocab, vocab_dict, fix_maxlen=qmaxlen)
        test_qsent.append(cut_sents)
        test_qlen.append(cut_len)
    if read_ans:
        test_a, _ = read_input(['../data/'+i+'/'+name+'.a' for i in dirs])
        test_asent = []
        test_alen = []
        for i in test_a:
            cut_sents, cut_len, vocab, vocab_dict = build_vocab(i, vocab, vocab_dict, fix_maxlen=amaxlen)
            test_asent.append(cut_sents)
            test_alen.append(cut_len)
    else:
        test_asent = None
        test_alen = None
    return test_qsent, test_qlen, test_asent, test_alen, vocab, vocab_dict

def cut_by_char(sent):
    re_hanzi = re.compile(r'(?P<hanzi>[\u4e00-\u9fa5\uff01-\uff5e])')
    re_space = re.compile(r' {2,}')
    return re_space.sub(' ', re_hanzi.sub(' \g<hanzi> ', sent)).strip(' ').split(' ')

def recut(sent, mvocab=None):
    global vocab
    if mvocab is None:
        mvocab = vocab
    return cut_by_char(''.join([mvocab[i] for i in sent[:(sent+[3]).index(3)]]))

def eval(tar, ref, vocab):
    tar = [recut(x[:(x+[3]).index(3)], vocab) for x in tar]
    ref = [recut(x[:(x+[3]).index(3)], vocab) for x in ref]
    return compute_bleu(tar, ref)[0]

def print_ans(sent, mvocab=None):
    global vocab
    if mvocab is None:
        mvocab = vocab
    return ' '.join([mvocab[i] for i in sent[:(sent+[3]).index(3)]])

def main():
    global vocab
    dirs = ['word_012', 'word_12', 'word_012_clf', 'word_12_clf', 'word_012_dclf', 'word_12_dclf']
    birnn = [True for _ in range(len(dirs))] + [False for _ in range(len(dirs))]
    layers = [2 for _ in range(len(dirs))] + [3 for _ in range(len(dirs))]
    dirs += dirs
    train_qsent, train_qlen, train_asent, train_alen, vocab, vocab_dict, qmaxlen, amaxlen = \
        read_trainset(dirs)
    dev_qsent, dev_qlen, dev_asent, dev_alen, vocab, vocab_dict = \
        read_dev_test_set('dev', dirs, vocab, vocab_dict, qmaxlen, amaxlen)
    test_qsent, test_qlen, test_asent, test_alen, vocab, vocab_dict = \
        read_dev_test_set('test', dirs, vocab, vocab_dict, qmaxlen, amaxlen)
    w2v = build_vocab_matrix(vocab, vocab_dict)
    model = Model(len(dirs), np.array(w2v), qmaxlen, amaxlen, learning_rate=0.001, n_hidden=256, layers=layers, birnn=birnn, copynet=False, adam=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    timestr = str(int(time.time()))
    max_bleu = [0. for _ in range(len(dirs))]
    max_step = [0 for _ in range(len(dirs))]
    # states = [False for _ in range(6)] + [True for _ in range(6)]
    states = None
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            print('train...')
            model.train_all(sess, train_qsent, train_qlen, train_asent[0], train_alen[0], states=states)
            # model.train(sess, 0, train_qsent[0], train_qlen[0], train_asent[0], train_alen[0])
            print_id = np.random.randint(len(test_qsent[0]))
            print('src:', print_ans(test_qsent[0][print_id], vocab))
            print('tar:', print_ans(test_asent[0][print_id], vocab))
            for j in range(len(dirs)):
            # for j in range(1):
                drt = model.test_sep(sess, j, dev_qsent[j], dev_qlen[j])
                trt = model.test_sep(sess, j, test_qsent[j], test_qlen[j])
                dev_score = int(eval(dev_asent[0], drt, vocab)*10000)/100
                test_score = int(eval(test_asent[0], trt, vocab)*10000)/100
                print('Model %d: pre:'%j, print_ans(trt[print_id], vocab), 'dev...bleu=', dev_score, '; test...bleu=', test_score)
                if dev_score > max_bleu[j]:
                    max_bleu[j] = dev_score
                    max_step[j] = i
                    model.save_model(sess, '../checkpoints/12_256_word_sep_model_'+timestr+'_%d'%j, i, '_l%d'%j)
            
    # prop = [i-0.2 for i in max_bleu]
    model.build_ml_decoder(models=[6,7,8,9,11], name='6-10,12')
    timestr = '1523437524'
    max_step = [5, 10, 5, 6, 7, 9, 7, 4, 6, 7, 8, 6]
    max_bleu = [34.85, 36.29, 36.06, 36.19, 36.7, 37.68, 38.34, 39.26, 39.1, 38.7, 36.87, 38.86]
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(len(dirs)):
            model.restore_model(sess, '../checkpoints/12_sep_model_'+timestr+'_%d'%i, max_step[i], '_l%d'%i)

        for i in range(100):
            drt = model.test(sess, dev_qsent, dev_qlen)
            trt = model.test(sess, test_qsent, test_qlen)
            dev_score = int(eval(dev_asent[0], drt[0])*10000)/100
            test_score = int(eval(test_asent[0], trt[0])*10000)/100
            print('dev...bleu=', dev_score, '; test...bleu=', test_score)
            print_id = np.random.randint(len(test_qsent[0]))
            print('src:', print_ans(test_qsent[0][print_id], vocab))
            print('tar:', print_ans(test_asent[0][print_id], vocab))
            print('pre:', print_ans(trt[0][print_id], vocab))
            print('train...(step2)')
            print(model.train_ml(sess, train_qsent, train_qlen, train_asent[0], train_alen[0]))