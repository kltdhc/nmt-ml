import numpy as np
from model import model as Model
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu

def bleu4(ref, cand):
    ref = [ref]
    cand = cand
    s = sentence_bleu(ref, cand, weights=(0, 0, 0, 1))
    if s>0.995 and sentence_bleu(ref, cand, weights=(1, 0, 0, 0))<0.9:
        s = 0
    return s

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
        cut_sent_num = [vocab_dict['<s>']]
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
            cut_sents[i] += [vocab_dict['<space>'] for _ in range(maxlen - len(cut_sents[i]))]
        if len(cut_sents[i])>maxlen:
            cut_len[-1] = maxlen
            cut_sents[i] = cut_sents[i][:maxlen]
    
    return cut_sents, cut_len, vocab, vocab_dict

def build_vocab_matrix(vocab, vocab_dict):
    infile = open('../data/char_emb', encoding='utf8')
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

def eval(tar, ref):
    count = 0
    for t, r in zip(tar, ref):
        if 3 in t:
            t_end = t.index(3)
        else:
            t_end = len(t)-1
        if 3 in r:
            r_end = r.index(3)
        else:
            r_end = len(t)-1
        count += bleu4(t[:t_end+1], r[:r_end+1])
    return count/len(tar)

def main():
    dirs = ['char_012', 'char_12', 'char_012_clf', 'char_12_clf', 'char_012_dclf', 'char_12_dclf']
    train_qsent, train_qlen, train_asent, train_alen, vocab, vocab_dict, qmaxlen, amaxlen = \
        read_trainset(dirs)
    dev_qsent, dev_qlen, dev_asent, dev_alen, vocab, vocab_dict = \
        read_dev_test_set('dev', dirs, vocab, vocab_dict, qmaxlen, amaxlen)
    test_qsent, test_qlen, test_asent, test_alen, vocab, vocab_dict = \
        read_dev_test_set('test', dirs, vocab, vocab_dict, qmaxlen, amaxlen)
    w2v = build_vocab_matrix(vocab, vocab_dict)
    model = Model(len(dirs), np.array(w2v), qmaxlen, amaxlen)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            model.train_all(sess, train_qsent, train_qlen, train_asent[0], train_alen[0])
            print('train...')
            rt = model.test(sess, dev_qsent, dev_qlen)
            print('test...bleu=', eval(test_asent[0], rt[0]))
            for j in range(6):
                rt = model.test_sep(sess, j, dev_qsent[j], dev_qlen[j])
                print('test...%d...bleu='%j, eval(test_asent[0], rt[0]))