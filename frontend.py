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
