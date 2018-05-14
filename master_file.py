"""
Author: Rafael Rivas
Date: 05/13/2018
Professor: Avner Biblarz
Title: masterfile.py
Abstract: Creates algorithm for a one week in advance price as well as
creates a long term prediction for the coin.
"""


import machine_learning.analysis as a
import twitter_api.top_crypto_tweets as ml
import requests, json

#Returns one week in advance predicted value off of current market climate
def algo() :
    # Gets Bitcoin price
    url = 'https://rest.coinapi.io/v1/exchangerate/BTC/USD'
    headers = {'X-CoinAPI-Key' : 'E8A5467E-9F0B-4A0E-80C3-8B7FA803E822'}
    response = requests.get(url, headers=headers)
    data = response.json()
    val = int(data['rate'])

    #Machine Learning Evaluates Tweets
    test = ml.returnBtcTweets()

    aftermath = []
    for i in test :
        aftermath.append(a.eval(i))

    positive = 0

    for r in aftermath :
        if r > 0 :
            positive += 1
    #Computes Prediciton Value
    val *= (1+((positive *2)-50)/100) / (3.14 - 1.07)
    return val

#Predicts major changes
def predict():
    test = ml.returnBtcTweets()

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

    prediction = abs(positive - negative)

    if prediction > 70:
        return "Major Increase in Price Incoming"
    elif prediction > 60:
        return "Minor Increase in Price Incoming"
    elif (prediction > 50) :
        return "Price will remain static"
    elif (prediction > 40) :
        return "Minor Decrease in Price Incoming"
    else :
        return "Major Decrease in Price Incoming"
