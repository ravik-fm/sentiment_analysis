import numpy as np

from keras.layers import Dense
from keras.layers import Conv1D, Dense, Input, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.regularizers import l2
from keras.models import Model
from keras.engine import Input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import keyedvectors
from gensim.models import word2vec
from collections import defaultdict
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
import sys
import os

def read_dir(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]

def read_file(filepath):

    with open(filepath, 'r') as myfile:
        str = myfile.read().replace('\n', '')
        return str

def process_text(text):
    """
    Process the input text by tokenizing and padding it.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    x_train = tokenizer.texts_to_sequences(text)

    x_train = pad_sequences(x_train, 500)
    return x_train

def read_training_data(files,label,texts,texts_w2v,labels):

    for file in files:
        str = read_file(file)
        texts.append(str)
        texts_w2v.append(str.split(' '))
        labels.append(label)


top_words = 10000
train_dir = sys.argv[1]+"/train"
test_dir = sys.argv[1]+"/test"

train_pos_files = read_dir(train_dir+"/pos")
train_neg_files = read_dir(train_dir+"/neg")

test_pos_files = read_dir(train_dir+"/pos")
test_neg_files = read_dir(train_dir+"/neg")

train_texts = []
texts_w2v = []
train_labels = []

read_training_data(train_pos_files,1,train_texts,texts_w2v,train_labels)
read_training_data(train_neg_files,0,train_texts,texts_w2v,train_labels)

test_texts = []
test_labels = []

read_training_data(test_pos_files,1,test_texts,texts_w2v,test_labels)
read_training_data(test_neg_files,0,test_texts,texts_w2v,test_labels)
print "Read training data"
print "Building word vectors"
Keras_w2v = word2vec.Word2Vec(min_count=3)
Keras_w2v.build_vocab(texts_w2v)
all_texts = train_texts + test_texts
Keras_w2v.train(all_texts, total_examples=Keras_w2v.corpus_count, epochs=Keras_w2v.iter)
Keras_w2v_wv = Keras_w2v.wv
embedding_layer = Keras_w2v_wv.get_keras_embedding()
print "Finished word2vec"
X_train = process_text(train_texts)
X_test = process_text(test_texts)

y_train =  to_categorical(np.asarray(train_labels))
y_test = to_categorical(np.asarray(test_labels))

print X_train.shape
print y_train.shape

print X_test.shape
print y_test.shape

sequence_input = Input(shape=(500,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(64, 3, activation='relu')(embedded_sequences)
x = Conv1D(32, 3, activation='relu')(x)
x = Conv1D(16, 3, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(180, activation='relu')(x)
preds = Dense(y_train.shape[1], activation='softmax')(x)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print "Training model"
model.fit(X_train, y_train, epochs=5)

eval = model.evaluate(X_test,y_test,batch_size=1000)

print eval
