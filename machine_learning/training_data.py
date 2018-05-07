import sys
import os

sys.path.insert(0, os.path.abspath("../data_processing"))

import tweet_processing as tp

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from bs4 import BeautifulSoup
import re

"""
cols = ['sentiment','id','date','query_string','user','text']
data = pd.read_csv("./training_data/training.1600000.processed.noemoticon.csv",header=None, names=cols, encoding="utf-8")

#drop useless column
data.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)

#add a column to check the text len before cleaning
data['pre_clean_len'] = [len(t) for t in data.text]

#ceate a dictionary for the data
data_dict = {
        'sentiment':{
            'type':data.sentiment.dtype,
            'description':'sentiment class - 0:negative, 1:positive'
        },
        'text':{
            'type':data.text.dtype,
            'description':'tweet text'
        },
        'pre_clean_len':{
            'type':data.pre_clean_len.dtype,
            'description':'Length of the tweet before cleaning'
        },
        'dataset_shape':data.shape
}

nums = [0, 400000, 800000, 1200000, 1600000]
print("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(nums[0], nums[4]):
    if ((i+1)%10000 == 0):
        print("Tweets %d of %d has been processed" % (i+1, nums[4]))

    clean_tweet_texts.append(tp.clean_tweet(data['text'][i]))

clean_data = pd.DataFrame(clean_tweet_texts, columns=['text'])
clean_data['target'] = data.sentiment

clean_data.to_csv('clean_tweet.csv', encoding='utf-8')
"""
csv = 'clean_tweet.csv'

my_data = pd.read_csv(csv, index_col = 0)
"""
my_data.dropna(inplace=True)
my_data.reset_index(drop=True, inplace=True)
my_data.to_csv('clean_tweet.csv', encoding='utf-8')
print(my_data.info())
"""
#pprint(data_dict)

"""
(fig, ax) = plt.subplots(figsize=(5,5))
plt.boxplot(data.pre_clean_len)
plt.show()
"""

#print(data.text[279])
#print(BeautifulSoup(data.text[279], 'lxml').get_text())

#print(data.text[343])
#print(re.sub(r'@[A-Za-z0-9]+', '', data.text[343]))

#print(data['pre_clean_len'])
#print(data.head(10))
#num = int(input("enter a number:"))

#print(data.text[num])
#s = tp.clean_tweet(data.text[num])
#for c in s:
#    print(ord(c))
#print(s)
