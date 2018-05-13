import machine_learning.analysis as a
import twitter_api.top_crypto_tweets as ml

test = ml.returnBtcTweets()

print(a.eval(test[0]))
