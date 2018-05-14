"""
Author: Rafael Rivas
Date: 05/13/2018
Professor: Avner Biblarz
Title: top_crypto_tweets.py
Abstract: Recieves 100 tweets from twitter in english that are popular and recent. Then compiles them into a lsit 
to be passed into the ml-eval function.
"""


import twitter

api = twitter.Api(consumer_key='zphUX92wy7r8iw5xMiWVIMAxn',
  consumer_secret='n38eNZylwwh5RM4pbP3WRoKN0dAlmg8Cv1aza8a2hvVLDevNfT',
  access_token_key='1856293255-UNGQNoLdq1BVGBUqN4ttSEWifGsbdG9NTHFL015',
  access_token_secret='SpjLStwyRWGCw5z4czj4FXCtzPA3UftnrNQ6Q7YUtRCmn')

def returnBtcTweets() :
    search = api.GetSearch("bitcoin", lang='en', count=100, result_type='mixed')
    top_100_tweets = []
    for t in search :
        top_100_tweets.append(t.text)
    return top_100_tweets
