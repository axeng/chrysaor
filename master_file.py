import machine_learning.analysis as a
import twitter_api.top_crypto_tweets as ml


while True:
    test = ml.returnBtcTweets()

    # print(a.eval(test[0]))

    aftermath = []
    for i in test :
        aftermath.append(a.eval(i))

    positive = 0
    negative = 0

    for r in aftermath :
        if r > 0 :
            positive += 1
        else :
            negative += 1
    print (positive, negative)

    wait = input("")
