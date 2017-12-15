import random
import spacy
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from time import time
from tqdm import tqdm

#from sklearn.naive_bayes import MultinomialNB
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string

try:
    import cPickle as pickle
except ImportError:
    import pickle

import dynet_config
dynet_config.set(mem=4096, random_seed=12345, autobatch=True) # was 2048 mem
dynet_config.set_gpu()

import dynet as dy

dyparams = dy.DynetParams()
#dyparams.init()
dyparams.set_requested_gpus(1)

import pdb


print("Loading spaCy...", end=' ')
nlp = spacy.load('en')
assert nlp.path is not None
print ('done.')


MAX_LEN = 100
NUM_TAGS = 3883
# VOCAB_CAP = 10000
VOCAB_CAP = 50000

UNK = '<UNK>'
START = '<S>'
END = '</S>'


# # Load Data for Parsing
print("Loading data...")

with open('/usr0/home/mamille2/twitter/data/huang2016/huang2016_train.aligned.pkl', 'rb') as f:
#     twitter_texts, twitter_tags, twitter_histories = pickle.load(f)
    twitter_texts, twitter_tags, _ = pickle.load(f)
#     , twitter_tags, _ = pickle.load(f)
    
with open('/usr0/home/mamille2/twitter/data/huang2016/huang2016_valid.aligned.pkl', 'rb') as f:
#     dev_texts, dev_tags, dev_histories = pickle.load(f)
    dev_texts, dev_tags, _ = pickle.load(f)
#     _, dev_tags, _ = pickle.load(f)
    
with open('/usr0/home/mamille2/twitter/data/huang2016/huang2016_test.aligned.pkl', 'rb') as f:
#     _, test_tags, _ = pickle.load(f)
    test_texts, test_tags, _ = pickle.load(f)
    

# # Load Data

def index_tags(tags_list, tag_set, tag_dict):
    return [[tag_dict[tag] for tag in tags if tag in tag_set] for tags in tags_list]


# Extract tag set
tag_counts = defaultdict(int)
for t in twitter_tags:
    for x in t:
        tag_counts[x] += 1

top_k_tags = set(sorted(tag_counts, key=tag_counts.get, reverse=True)[:NUM_TAGS])

tag_set = set()
for t in twitter_tags:
    tag_set.update(set([x for x in t if x in top_k_tags]))
    
tag_set = sorted(tag_set)
print ('{} unique tags.'.format(len(tag_set)))

tag_indexes = defaultdict(lambda: len(tag_indexes))
parsed_tags = index_tags(twitter_tags, tag_set, tag_indexes)
idx_to_tag = {v: k for k, v in tag_indexes.items()}


try:
    print ('Attempting to open preprecessed TRAIN data ... ', end='')
#     raise NotImplemented
    
    t0=time()
    with open('/usr0/home/mamille2/twitter/data/huang2016/parsed_twitter_train_data_no_histories.pkl', 'rb') as f:
        vocab, parsed_texts, parsed_tags = pickle.load(f)
    print ('DONE. ({:.3f}s)'.format(time()-t0))
        
except:
    print ('FAIL.')
    
    print ('\tParsing texts ... ', end='')
    t0=time()
    parsed_texts = [[str(w) for w in t][:MAX_LEN] for t in nlp.pipe([x.encode('ascii', 'ignore').decode('ascii').lower() for x in twitter_texts], n_threads=3, batch_size=20000)]
    print ('DONE. ({:.3f}s)'.format(time()-t0))
    
    print ('\tCounting words ... ', end='')
    word_counts = defaultdict(int)
    for t in parsed_texts:
        for x in t:
            word_counts[x] += 1
    top_k_words = set(sorted(word_counts, key=word_counts.get, reverse=True)[:VOCAB_CAP-3])

    word_set = set()
    for t in parsed_texts:
        word_set.update(set([x for x in t if x in top_k_words]))
    print ('DONE. ({:.3f}s)'.format(time()-t0)) 
    
    vocab = defaultdict(lambda: len(vocab))
    print ('\tIndexing texts ... ', end='')
    t0=time()
    parsed_texts = [[vocab[START]] + [(vocab[w] if w in word_set else vocab[UNK]) for w in t] + [vocab[END]] for t in parsed_texts]
    print ('DONE. ({:.3f}s)'.format(time()-t0))
    
    unk_idx = vocab[UNK]
    sos_idx = vocab[START]
    eos_idx = vocab[END]
    
    print ('\tSAVING parsed data ... ', end='')
    t0=time()
    with open('parsed_twitter_train_data_no_histories.pkl', 'wb') as f:
        pickle.dump((dict(vocab), parsed_texts, parsed_tags), f) 
    print ('DONE. ({:.3f}s)'.format(time()-t0))

unk_idx = vocab[UNK]
sos_idx = vocab[START]
eos_idx = vocab[END]
# Set unknown words to be UNK --> note as written, the paper does not indicate that any training data is labeled as UNK...
vocab = defaultdict(lambda: unk_idx, vocab)
idx_to_vocab = {v: k for k, v in vocab.items()}

VOCAB_SIZE = len(vocab)
print ('Vocab size:', VOCAB_SIZE)


# Check number of tags
#tagc = Counter([t for tags in parsed_tags for t in tags])
#len(tagc)


try:
    print ('Attempting to open preprecessed DEV and TEST data ... ', end='')
#     raise NotImplemented
    
    t0=time()
    with open('/usr0/home/mamille2/twitter/data/huang2016/parsed_twitter_test_dev_data_no_histories.pkl', 'rb') as f:
        parsed_dev_texts, parsed_test_texts = pickle.load(f)
    print ('DONE. ({:.3f}s)'.format(time()-t0))
        
except:
    print ('FAIL.')
    print ('\tParsing texts ... ', end='')
    t0=time()
    parsed_dev_texts = [[vocab[START]] + [vocab[str(w)] for w in t if not w.is_stop][:MAX_LEN] + [vocab[END]] for t in nlp.pipe([x.encode('ascii', 'ignore').decode('ascii').lower() for x in dev_texts], n_threads=3, batch_size=20000)]
    parsed_test_texts = [[vocab[START]] + [vocab[str(w)] for w in t if not w.is_stop][:MAX_LEN] + [vocab[END]] for t in nlp.pipe([x.encode('ascii', 'ignore').decode('ascii').lower() for x in test_texts], n_threads=3, batch_size=20000)]
    print ('DONE. ({:.3f}s)'.format(time()-t0))
    
    print ('\tSAVING parsed data ... ', end='')
    t0=time()
    with open('parsed_twitter_test_dev_data_no_histories.pkl', 'wb') as f:
        pickle.dump((parsed_dev_texts, parsed_test_texts), f) 
    print ('DONE. ({:.3f}s)'.format(time()-t0))


train = list(zip(parsed_texts, parsed_tags))
dev_tags = index_tags(dev_tags, tag_set, tag_indexes)
dev = list(zip(parsed_dev_texts, dev_tags))
test_tags = index_tags(test_tags, tag_set, tag_indexes)
test = list(zip(parsed_test_texts, test_tags))


# # Model Parameters and Settings
print("Building model...")

EMBEDDING_DIM = 128
# HIDDEN_DIM = 256
HIDDEN_DIM = 512
Q_DIM = 512
DROPOUT = 0.2
# DROPOUT = 0
ALPHA = 0.01
EPSILON_MAX = .9
EPSILON_MIN = 0.00
KL_WEIGHT_START = 0.0

BATCH_SIZE = 16
PATIENCE = 3


# Initialize dynet model
model = dy.ParameterCollection()

# The paper uses AdaGrad
trainer = dy.AdamTrainer(model)

# Embedding parameters
embed = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDING_DIM))

# Recurrent layers for tweet encoding
lstm_encode = dy.LSTMBuilder(1, EMBEDDING_DIM, HIDDEN_DIM, model)
lstm_decode = dy.LSTMBuilder(1, EMBEDDING_DIM, Q_DIM, model)

# Encoder MLP for tweet encoding
W_mu_tweet_p = model.add_parameters((Q_DIM, HIDDEN_DIM))
V_mu_tweet_p = model.add_parameters((HIDDEN_DIM, Q_DIM))
b_mu_tweet_p = model.add_parameters((Q_DIM))

W_sig_tweet_p = model.add_parameters((Q_DIM, HIDDEN_DIM))
V_sig_tweet_p = model.add_parameters((HIDDEN_DIM, Q_DIM))
b_sig_tweet_p = model.add_parameters((Q_DIM))

W_mu_tag_p = model.add_parameters((Q_DIM, NUM_TAGS))
V_mu_tag_p = model.add_parameters((HIDDEN_DIM, Q_DIM))
b_mu_tag_p = model.add_parameters((Q_DIM))

W_sig_tag_p = model.add_parameters((Q_DIM, NUM_TAGS))
V_sig_tag_p = model.add_parameters((HIDDEN_DIM, Q_DIM))
b_sig_tag_p = model.add_parameters((Q_DIM))

W_mu_p = model.add_parameters((Q_DIM, 2 * HIDDEN_DIM))
b_mu_p = model.add_parameters((Q_DIM))

W_sig_p = model.add_parameters((Q_DIM, 2 * HIDDEN_DIM))
b_sig_p = model.add_parameters((Q_DIM))

W_hidden_p = model.add_parameters((HIDDEN_DIM, Q_DIM))
b_hidden_p = model.add_parameters((HIDDEN_DIM))

W_tweet_softmax_p = model.add_parameters((VOCAB_SIZE, Q_DIM))
b_tweet_softmax_p = model.add_parameters((VOCAB_SIZE))

W_tag_output_p = model.add_parameters((NUM_TAGS, HIDDEN_DIM))
b_tag_output_p = model.add_parameters((NUM_TAGS))



def reparameterize(mu, log_sigma_squared):
    d = mu.dim()[0][0]
    sample = dy.random_normal(d)
    covar = dy.exp(log_sigma_squared * 0.5)

    return mu + dy.cmult(covar, sample)

def mlp(x, W, V, b):
    return V * dy.tanh(W * x + b)


def calc_loss(sent, epsilon=0.0):
    #dy.renew_cg()
    
    # Transduce all batch elements with an LSTM
    src = sent[0]
    tags = sent[1]

    # initialize the LSTM
    init_state_src = lstm_encode.initial_state()

    # get the output of the first LSTM
    src_output = init_state_src.add_inputs([embed[x] for x in src])[-1].output()

    # Now compute mean and standard deviation of source hidden state.
    W_mu_tweet = dy.parameter(W_mu_tweet_p)
    V_mu_tweet = dy.parameter(V_mu_tweet_p)
    b_mu_tweet = dy.parameter(b_mu_tweet_p)

    W_sig_tweet = dy.parameter(W_sig_tweet_p)
    V_sig_tweet = dy.parameter(V_sig_tweet_p)
    b_sig_tweet = dy.parameter(b_sig_tweet_p)
    
    # Compute tweet encoding
    mu_tweet      = dy.dropout(mlp(src_output, W_mu_tweet,  V_mu_tweet,  b_mu_tweet), DROPOUT)
    log_var_tweet = dy.dropout(mlp(src_output, W_sig_tweet, V_sig_tweet, b_sig_tweet), DROPOUT)
    
    W_mu_tag = dy.parameter(W_mu_tag_p)
    V_mu_tag = dy.parameter(V_mu_tag_p)
    b_mu_tag = dy.parameter(b_mu_tag_p)

    W_sig_tag = dy.parameter(W_sig_tag_p)
    V_sig_tag = dy.parameter(V_sig_tag_p)
    b_sig_tag = dy.parameter(b_sig_tag_p)
    
    # Compute tag encoding
    tags_tensor = dy.sparse_inputTensor([tags], np.ones((len(tags),)), (NUM_TAGS,))
    
    mu_tag      = dy.dropout(mlp(tags_tensor, W_mu_tag,  V_mu_tag,  b_mu_tag), DROPOUT)
    log_var_tag = dy.dropout(mlp(tags_tensor, W_sig_tag, V_sig_tag, b_sig_tag), DROPOUT)
    
    # Combine encodings for mean and diagonal covariance
    W_mu = dy.parameter(W_mu_p)
    b_mu = dy.parameter(b_mu_p)

    W_sig = dy.parameter(W_sig_p)
    b_sig = dy.parameter(b_sig_p)
    
    # Slowly phase out getting both inputs
    if random.random() < epsilon:
        mask = dy.zeros(HIDDEN_DIM)
    else:
        mask = dy.ones(HIDDEN_DIM)
        
    if random.random() < 0.5:
        mu_tweet = dy.cmult(mu_tweet, mask)
        log_var_tweet = dy.cmult(log_var_tweet, mask)
    else:
        mu_tag = dy.cmult(mu_tag, mask)
        log_var_tag = dy.cmult(log_var_tag, mask)
    
    mu      = dy.affine_transform([b_mu,  W_mu,  dy.concatenate([mu_tweet, mu_tag])])
    log_var = dy.affine_transform([b_sig, W_sig, dy.concatenate([log_var_tweet, log_var_tag])])

    # KL-Divergence loss computation
    #kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    z = reparameterize(mu, log_var)

    # now step through the output sentence
    all_losses = []

    current_state = lstm_decode.initial_state().set_s([z, dy.tanh(z)])
    prev_word = src[0]
    W_sm = dy.parameter(W_tweet_softmax_p)
    b_sm = dy.parameter(b_tweet_softmax_p)

    for next_word in src[1:]:
        # feed the current state into the
        
        current_state = current_state.add_input(embed[prev_word])
        output_embedding = current_state.output()

        s = dy.affine_transform([b_sm, W_sm, output_embedding])
        
        all_losses.append(dy.pickneglogsoftmax(s, next_word))

        # Slowly phase out teacher forcing (this may be slow??)
        if random.random() < epsilon:
            p = dy.softmax(s).npvalue()
            prev_word = np.random.choice(VOCAB_SIZE, p=p/p.sum())
        else:
            prev_word = next_word
    
    softmax_loss = dy.esum(all_losses)

    W_hidden = dy.parameter(W_hidden_p)
    b_hidden = dy.parameter(b_hidden_p)
    
    W_out = dy.parameter(W_tag_output_p)
    b_out = dy.parameter(b_tag_output_p)
    
    h = dy.dropout(dy.tanh(b_hidden + W_hidden * z), DROPOUT)
    o = dy.logistic(b_out + W_out * h)
    
    crossentropy_loss = dy.binary_log_loss(o, tags_tensor)
                               
    #return kl_loss, softmax_loss, crossentropy_loss
    return softmax_loss, crossentropy_loss



# Training
#print("Training...")
print ('Using batch size of {}.'.format(BATCH_SIZE))

epsilon = EPSILON_MIN
#kl_weight = KL_WEIGHT_START
steps = 0
strikes = 0
last_dev_loss = np.inf
for ITER in range(100):
    # Perform training
    random.shuffle(train)
    
    batches = [train[i:i + BATCH_SIZE] for i in range(0, len(train), BATCH_SIZE)]
    
    #train_words, train_loss, train_kl_loss, train_reconstruct_loss, total_tag_loss = 0, 0.0, 0.0, 0.0, 0.0
    train_words, train_loss, train_reconstruct_loss, total_tag_loss = 0, 0.0, 0.0, 0.0
    start = time()
    
    print()
    print ('Training ... Iteration:', ITER, 'Epsilon:', epsilon)
    for i, batch in enumerate(tqdm(batches)):
        dy.renew_cg()
        #dy.renew_cg(immediate_compute=True, check_validity=True) # makes program die--is_valid() not implemented for CUDA yet
        #dy.renew_cg(immediate_compute=True)
        losses = []
        for sent_id, sent in enumerate(batch):
            if len(sent[1]) < 1 or len(sent[0]) < 3:
                continue
            #kl_loss, softmax_loss, tag_loss = calc_loss(sent, epsilon)
            softmax_loss, tag_loss = calc_loss(sent, epsilon)
            #total_loss = dy.esum([kl_loss, softmax_loss, tag_loss])
            #train_loss += total_loss.value()
            
            # Gradually increase KL-Divergence loss
#             if steps < 15000:
#                 kl_weight = 1 / (1 + np.exp(-0.001 * steps + 5))
#             else:
#                 kl_weight = 1.0

            # Zero out KL weight
            #kl_weight = 0.0
            #kl_weight = 1e-6
                
            #losses.append(dy.esum([kl_weight * kl_loss, softmax_loss, tag_loss]))
            losses.append(dy.esum([softmax_loss, tag_loss])) # fails with gradient nan error

            # Record the KL loss and reconstruction loss separately help you monitor the training.
            #train_kl_loss += kl_loss.value()
            train_reconstruct_loss += softmax_loss.value()
            total_tag_loss += tag_loss.value()
            
            train_words += len(sent[0])
        steps += 1
   
        # Batch update
        batch_loss = dy.esum(losses)/BATCH_SIZE
        train_loss += batch_loss.value()
        batch_loss.backward()
        #batch_loss.backward(full=True)
        pdb.set_trace()
        trainer.update() # fails here with gradient nan or inf error
        
        
        #total_loss.backward()
        #trainer.update()
        #if (sent_id + 1) % 1000 == 0:

        #    print("--finished %r sentences" % (sent_id + 1))

    # Gradually increase KL-Divergence loss
    if steps < 100000:
        epsilon = .9 / (1 + np.exp(-0.0001 * steps + 5))
    else:
        epsilon = EPSILON_MAX
        
    #epsilon = min(EPSILON_MAX, epsilon + 0.05)
    try:
        train_ppl = math.exp(train_loss / train_words)
    except OverflowError as e:
        train_ppl = 1e6

#    print("iter %r: train loss/word=%.4f, kl loss/word=%.4f, reconstruction loss/word=%.4f, ppl=%.4f, tag loss=%.4f" % (
#        ITER, train_loss / train_words, train_kl_loss / train_words, train_reconstruct_loss / train_words,
#        train_ppl, total_tag_loss / len(train)))

    print("iter %r: train loss/word=%.4f, reconstruction loss/word=%.4f, ppl=%.4f, tag loss=%.4f" % (
        ITER, train_loss / train_words, train_reconstruct_loss / train_words,
        train_ppl, total_tag_loss / len(train)))

    # Evaluate on dev set
    #dev_words, dev_loss, dev_kl_loss, dev_reconstruct_loss, dev_tag_loss = 0, 0.0, 0.0, 0.0, 0.0
    dev_words, dev_loss, dev_reconstruct_loss, dev_tag_loss = 0, 0.0, 0.0, 0.0
    start = time()
    print ('Evaluating batch ... ')
    for sent_id, sent in enumerate(tqdm(dev)):
        dy.renew_cg()
        if len(sent[1]) < 1 or len(sent[0]) < 3:
                continue
        #kl_loss, softmax_loss, tag_loss = calc_loss(sent)
        softmax_loss, tag_loss = calc_loss(sent)

        #dev_kl_loss += kl_loss.value()
        dev_reconstruct_loss += softmax_loss.value()
        dev_tag_loss += tag_loss.value()
        #dev_loss += kl_loss.value() + softmax_loss.value() + tag_loss.value()
        dev_loss += softmax_loss.value() + tag_loss.value() # take out kl_loss

        dev_words += len(sent[0])
        trainer.update()

    try:
        dev_ppl = math.exp(dev_loss / dev_words)
    except:
        dev_ppl = 1e6

#    print("iter %r: dev loss/word=%.4f, kl loss/word=%.4f, reconstruction loss/word=%.4f, ppl=%.4f, tag loss=%.2f" % (
#        ITER, dev_loss / dev_words, dev_kl_loss / dev_words, dev_reconstruct_loss / dev_words,
#        dev_ppl, dev_tag_loss / len(dev)))

    print("iter %r: dev loss/word=%.4f, reconstruction loss/word=%.4f, ppl=%.4f, tag loss=%.2f" % (
        ITER, dev_loss / dev_words, dev_reconstruct_loss / dev_words,
        dev_ppl, dev_tag_loss / len(dev)))

    if dev_loss > last_dev_loss and ITER > 9:
        strikes += 1
    else:
        strikes = 0
        last_dev_loss = dev_loss
        model.save('/usr0/home/mamille2/twitter/models/tweet_tag_vae.best.weights')
        
    if strikes >= PATIENCE:
        print ('Early stopping after {} iterations.')
        break


try:
	outpath = '/usr0/home/mamille2/twitter/models/trained_project.weights'
	print("Saving model weights to {}...".format(outpath), end=' ')
	model.save(outpath)
	print("done.")
except:
	pdb.set_trace()
