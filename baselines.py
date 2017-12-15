import itertools
import spacy
import pickle
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras.backend as K
import tensorflow as tf

from collections import defaultdict, Counter

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Embedding, Dropout, Reshape, TimeDistributed
from keras.utils import to_categorical, plot_model

from keras_tqdm import TQDMNotebookCallback, TQDMCallback

import tensorflow as tf

from pprint import pprint

from scipy import stats
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from tqdm import tqdm


# Check if GPU enabled

#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()


# # Data Processing

# ## Twitter

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



# Extract character set

chars = set()
for text in twitter_texts:
    chars.update(set(text))
    
chars = sorted(chars)
chars = ['<PAD>', '<UNK>'] + chars
print ('{} unique characters.'.format(len(chars)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


MAX_LENGTH = 140


print("Vectorizing data...")
# Vectorize tweets

# train
train_X = np.zeros((len(twitter_texts), MAX_LENGTH), dtype=np.bool)
for i, text in tqdm(enumerate(twitter_texts), total=len(twitter_texts)):
    for t, char in enumerate(text):
        if t >= MAX_LENGTH:
            break
        train_X[i, t] = char_indices[char]
        
# dev
dev_X = np.zeros((len(dev_texts), MAX_LENGTH), dtype=np.bool)
for i, text in tqdm(enumerate(dev_texts), total=len(dev_texts)):
    for t, char in enumerate(text):
        if t >= MAX_LENGTH:
            break
        dev_X[i, t] = char_indices.get(char, '<UNK>')
        
# test
test_X = np.zeros((len(test_texts), MAX_LENGTH), dtype=np.bool)
for i, text in tqdm(enumerate(test_texts), total=len(test_texts)):
    for t, char in enumerate(text):
        if t >= MAX_LENGTH:
            break
        test_X[i, t] = char_indices.get(char, '<UNK>')


# Extract tag set

tag_set = set()
for t in twitter_tags:
    tag_set.update(set(t))
    
tag_set = sorted(tag_set)
print ('{} unique tags.'.format(len(tag_set)))

# Select top 3883 tags like Huang+ 2016
twitter_tags_c = Counter([t for ts in twitter_tags for t in ts]).most_common(3883)
top_tags = set([t for t,_ in twitter_tags_c])

tag_indices = dict((t, i) for i, t in enumerate(top_tags))
indices_tag = dict((i, t) for i, t in enumerate(top_tags))
print ('Selected {} tags.'.format(len(top_tags)))


# Vectorize tags

# train
train_y = np.zeros((len(twitter_tags), len(top_tags)), dtype=np.bool)
for i, tags in tqdm(enumerate(twitter_tags), total=len(twitter_tags)):
    for tag in tags:
        if tag in top_tags:
            train_y[i, tag_indices[tag]] = 1
        
# dev
dev_y = np.zeros((len(dev_tags), len(top_tags)), dtype=np.bool)
for i, tags in tqdm(enumerate(dev_tags), total=len(dev_tags)):
    for tag in tags:
        if tag in top_tags:
            dev_y[i, tag_indices[tag]] = 1
            
# test
test_y = np.zeros((len(test_tags), len(top_tags)), dtype=np.bool)
for i, tags in tqdm(enumerate(test_tags), total=len(test_tags)):
    for tag in tags:
        if tag in top_tags:
            test_y[i, tag_indices[tag]] = 1



# # Build Model

# ## BiLSTM Baseline

print("Building model...")

#with tf.device('/cpu:0'):
model = Sequential()
model.add(Embedding(len(chars) + 1,
		    64,
		    input_length=MAX_LENGTH,
		    trainable=True,
		    mask_zero=True))
#model.add(Bidirectional(LSTM(128))) # segfault
model.add(Bidirectional(LSTM(32))) # segfault if 64 or higher
model.add(Dropout(.5))
model.add(Dense(len(top_tags), activation='softmax'))


model.compile(loss='categorical_crossentropy',
	      optimizer='adam',
	      metrics=['categorical_accuracy'])


filepath="~/twitter/models/baseline_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=3)
callbacks_list = [checkpoint, early_stop]

print("Model output to {}".format(filepath))

print("Training model...")
model.fit(train_X,
	      train_y,
	      validation_data=(dev_X, dev_y),
	      shuffle=True,
	      batch_size=32,
	      epochs=10,
	      verbose=True,
	      callbacks=callbacks_list)

print('BiLSTM Baseline Training complete!')
