"""
Author: Axen Georget
Date: 05/13/2018
Professor: Avner Biblarz
Title: analysis.py
Abstract: file containing analysis functions
"""
from keras.models import load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle

import data_processing.tweet_processing as tp

#Load the model of the neural network
loaded_CNN_model = load_model('machine_learning/CNN_best_weights.hdf5')

#Open the tokenizer file, saved to gain some time
handle = open('machine_learning/tokenizer.pickle', 'rb')
tokenizer = pickle.load(handle)

#Return if a tweet is positive or negative
def eval(tweet):
	#Clean the tweet
	tweet_cleaned = tp.clean_tweet(tweet)
	if tweet_cleaned == "":
		return -1

	#Vectorize the string for the neural network
	sequences_val = tokenizer.texts_to_sequences([tweet_cleaned])
	x_val_seq = pad_sequences(sequences_val, maxlen=45)

	#Return the output of the neural network
	return round(loaded_CNN_model.predict(x_val_seq)[0][0])
"""
print(eval("this week is not going as had hoped"))
print(eval("sad that the feet of my macbook just fell off"))
print(eval("another commenting contest yay"))
print(eval("i want to cry a lot"))
print(eval("My dad just gave me this new object, I'm so happy rn"))
"""
