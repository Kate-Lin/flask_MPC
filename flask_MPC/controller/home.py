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
    return render_template('MPC_HOME.html')


@app.route('/evoting')
def evoting():
    bullet = bulletin.query.all()
    if bullet == []:
        remain = 0
    elif bullet[0].voter_num == bullet[0].T:
        remain = 0
    else:
        remain = bullet[0].voter_num-bullet[0].T
    return render_template('EVOTING_INDEX.html',remain=remain)


@app.route('/AI')
def AI():
    return render_template('AI_INDEX.html')

@app.route('/AI/back_train')
def back_train():
    return render_template('AI_TRAIN.html')

@app.route('/AI/back_test')
def back_test():
    return render_template('AI_TEST_BACK.html')

@app.route('/evoting/error')
def error():
    return 'ERROR!!'


@app.route('/success')
def success():
    return 'SUCCESS!!'
