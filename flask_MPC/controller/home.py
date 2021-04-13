from flask_MPC.models.voter import voter
from flask_MPC.models.bulletin import bulletin
from flask_MPC.models.enc_vote import enc_vote
from flask_MPC.models.enc_sum import enc_sum
import numpy as np
from flask_MPC import app,db
from flask import request,render_template,flash,abort,url_for,redirect,session,Flask,g
from flask_MPC.controller.generate_voters import generate_voter_ID
from math import log2,floor
from torch import randperm
import random


@app.route('/')
def show_home():
    return render_template('home.html')


@app.route('/evoting')
def evoting():
    return render_template('evoting_home.html')


@app.route('/AI')
def AI():
    return render_template('AI_home.html')

@app.route('/AI/local_test')
def show_local():
    return render_template('test_local_data.html')


@app.route('/evoting/error')
def error():
    return 'ERROR!!'


@app.route('/success')
def success():
    return 'SUCCESS!!'
