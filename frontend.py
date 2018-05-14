#Author: Nicholas Cerda
#Date: 05/13/2018
#Professor: Avner Biblarz
#Title: frontend.py
#Abstract: File creates the webpage
import sys
import os


from flask import Flask, render_template
from flask_bootstrap import Bootstrap

sys.path.append(os.path.abspath("."))
import master_file as mf

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/')
def home():

    return render_template('test.html', prediction=mf.predict(), algo=mf.algo(), btc= mf.btcPrice())

app.run(debug = False)
