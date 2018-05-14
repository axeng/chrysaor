"""
Author: Axen Georget
Date: 05/13/2018
Professor: Avner Biblarz
Title: learning.py
Abstract: file containing all the learning process for the neural network
"""
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

seed = 7

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model

from keras.callbacks import ModelCheckpoint
from keras.models import load_model


#Load the training_data
csv = 'training_data/clean_tweet.csv'
print("Loading data")
data = pd.read_csv(csv, index_col=0)
print("Data loaded")

data.dropna(inplace=True)
data.reset_index(drop=True,inplace=True)

x = data.text
y = data.target

#Spleet the data
SEED = 200
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

#Load the vectorize training data
model_ug_cbow = KeyedVectors.load('training_data/w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('training_data/w2v_model_ug_sg.word2vec')

embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
#print('Found %s word vectors.' % len(embeddings_index))

#Tokenize the data
print("Start tokenizer")
tokenizer = Tokenizer(num_words=100000)
print(".")
tokenizer.fit_on_texts(x_train)
print("..")
sequences = tokenizer.texts_to_sequences(x_train)
print("End tokenizer")
#print(len(tokenizer.word_index))

"""
for x in x_train[:5]:
    print(x)

print(sequences[:5])
"""

print("Start pad sequences")
x_train_seq = pad_sequences(sequences, maxlen=45)
print("End pad sequences")
#print('Shape of data tensor:', x_train_seq.shape)

#print(x_train_seq[:5])

print("Start val tokenizer")
sequences_val = tokenizer.texts_to_sequences(x_validation)
print("End val tokenizer")
print("Start val pad sequences")
x_val_seq = pad_sequences(sequences_val, maxlen=45)
print("End val pad sequences")

#Vectorize the data
num_words = 100000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(np.array_equal(embedding_matrix[6] ,embeddings_index.get('you')))

#Create a neural network and train it with the data
structure_test = Sequential()
e = Embedding(100000, 200, input_length=45)
structure_test.add(e)
structure_test.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
print(structure_test.summary())
tweet_input = Input(shape=(45,), dtype='int32')

tweet_encoder = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=True)(tweet_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1)(merged)
output = Activation('sigmoid')(merged)
model = Model(inputs=[tweet_input], outputs=[output])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#Save the neural network to this path
filepath="CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

print(model.fit(x_train_seq, y_train, batch_size=32, epochs=5, validation_data=(x_val_seq, y_validation), callbacks = [checkpoint]))
