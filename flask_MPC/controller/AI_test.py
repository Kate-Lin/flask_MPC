import numpy as np
from flask_MPC import app,db
from flask import request,render_template,flash,abort,url_for,redirect,session,Flask,g
import os

@app.route('/AI/score_result',methods=['POST'])
def calculate_score():
    form = request.form
    homo = request.form.getlist('homo')[0]
    dataset = request.form.getlist('dataset')[0]
    model = request.form.getlist('model')[0]
    str1='python flask_MPC/controller/'+homo+'/'+dataset+'_'+model+'.py'
    print(os.getcwd())
    os.system(str1)
    return 'success'

