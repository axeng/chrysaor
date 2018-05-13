from keras.models import load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle

"""
import sys
import os

sys.path.insert(0, os.path.abspath("../data_processing"))

import tweet_processing as tp
"""

loaded_CNN_model = load_model('CNN_best_weights.01-0.8292.hdf5')

handle = open('tokenizer.pickle', 'rb')
tokenizer = pickle.load(handle)

def eval(tweet):
	tweet_cleaned = tp.clean_tweet(tweet)
	if tweet_cleaned == "":
		return

	sequences_val = tokenizer.texts_to_sequences([tweet_cleaned])
	x_val_seq = pad_sequences(sequences_val, maxlen=45)

	return round(loaded_CNN_model.predict(x_val_seq)[0][0])

print(eval("this week is not going as had hoped"))
print(eval("sad that the feet of my macbook just fell off"))
print(eval("another commenting contest yay"))
print(eval("i want to cry a lot"))
print(eval("My dad just gave me this new object, I'm so happy rn"))
