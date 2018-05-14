#Author: Nicholas Cerda
#Date: 05/13/2018
#Professor: Avner Biblarz
#Title: frontend.py
#Abstract: File creates the webpage
from flask import Flask, render_template, flash, redirect
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from datetime import datetime

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/')
def home():
    return render_template('test.html')
