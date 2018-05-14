"""
Author: Axen Georget
Date: 05/13/2018
Professor: Avner Biblarz
Title: tweet_processing.py
Abstract: file containing the tweet processing functions
"""
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer

tok = WordPunctTokenizer()
negations_dic = {
                    "isn't":"is not", 
                    "aren't":"are not", 
                    "wasn't":"was not", 
                    "weren't":"were not",
                    "haven't":"have not",
                    "hasn't":"has not",
                    "hadn't":"had not",
                    "won't":"will not",
                    "wouldn't":"would not",
                    "don't":"do not",
                    "doesn't":"does not",
                    "didn't":"did not",
                    "can't":"can not",
                    "couldn't":"could not",
                    "shouldn't":"should not",
                    "mightn't":"might not",
                    "mustn't":"must not"
                }

#Create the negation pattern
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

#Clean a tweet
def clean_tweet(tweet):
    #transform html
    text = BeautifulSoup(tweet, 'lxml').get_text()
    
    cleantext = ""

    #clean utf 8 chars
    for i in range(len(text)):
        if ord(text[i]) < 0 or ord(text[i]) > 127:
            cleantext += '?'
        else:
            cleantext += text[i]

    text = cleantext

    #clean users and links
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'https?://[^ ]+', '', text)
    text = re.sub(r'www.[^ ]+', '', text)

    #remove upper-cases characters
    text = text.lower()

    #change negations
    text = neg_pattern.sub(lambda l: negations_dic[l.group()], text)

    #clean useless chars
    text = re.sub("[^a-zA-Z]", " ", text)

    #remove unneccessary white spaces
    words = [x for x  in tok.tokenize(text) if len(x) > 1]

    return (" ".join(words)).strip()
    
