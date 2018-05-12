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



csv = 'training_data/clean_tweet.csv'
print("Loading data")
data = pd.read_csv(csv, index_col=0)
print("Data loaded")

data.dropna(inplace=True)
data.reset_index(drop=True,inplace=True)

x = data.text
y = data.target

SEED = 200
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)


model_ug_cbow = KeyedVectors.load('training_data/w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('training_data/w2v_model_ug_sg.word2vec')

embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
#print('Found %s word vectors.' % len(embeddings_index))

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

num_words = 100000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(np.array_equal(embedding_matrix[6] ,embeddings_index.get('you')))

"""
model_ptw2v = Sequential()
e = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=False)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)
"""
"""
model_ptw2v = Sequential()
e = Embedding(100000, 200, input_length=45)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)

"""
"""
model_ptw2v = Sequential()
e = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=True)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)
"""

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

filepath="CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

print(model.fit(x_train_seq, y_train, batch_size=32, epochs=5, validation_data=(x_val_seq, y_validation), callbacks = [checkpoint]))

"""
loaded_CNN_model = load_model('CNN_best_weights.02-0.8333.hdf5')
#loaded_CNN_model.evaluate(x=x_val_seq, y=y_validation)


tvec = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
tvec.fit(x_train)

x_train_tfidf = tvec.transform(x_train)
x_test_tfidf = tvec.transform(x_test)

lr_with_tfidf = LogisticRegression()
print(lr_with_tfidf.fit(x_train_tfidf,y_train))

print(lr_with_tfidf.score(x_test_tfidf,y_test))

yhat_lr = lr_with_tfidf.predict_proba(x_test_tfidf)

sequences_test = tokenizer.texts_to_sequences(x_test)
x_test_seq = pad_sequences(sequences_test, maxlen=45)

print(loaded_CNN_model.evaluate(x=x_test_seq, y=y_test))

yhat_cnn = loaded_CNN_model.predict(x_test_seq)




fpr, tpr, threshold = roc_curve(y_test, yhat_lr[:,1])
roc_auc = auc(fpr, tpr)
fpr_cnn, tpr_cnn, threshold = roc_curve(y_test, yhat_cnn)
roc_auc_nn = auc(fpr_cnn, tpr_cnn)
plt.figure(figsize=(8,7))
plt.plot(fpr, tpr, label='tfidf-logit (area = %0.3f)' % roc_auc, linewidth=2)
plt.plot(fpr_cnn, tpr_cnn, label='w2v-CNN (area = %0.3f)' % roc_auc_nn, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic: is positive', fontsize=18)
plt.legend(loc="lower right")
plt.show()
"""
