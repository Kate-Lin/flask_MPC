import numpy as np
from flask_MPC import app,db
from flask import request,render_template,flash,abort,url_for,redirect,session,Flask,g
import os
from joblib import dump,load
from ckks.CKKS_Alice import CKKS_Alice
from ckks.CKKS_Bob import CKKS_Bob
from phe.Phe_Alice import Phe_Alice
from phe.Phe_Bob import Phe_Bob
import math
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,\
    roc_curve,classification_report,mean_squared_error, mean_squared_log_error,r2_score
from util.conf_matrix import draw_conf_matrix
from util.roc import draw_ROC_curve
import time

def find_next_power(x):
    return 2 ** math.ceil(math.log2(x))


def choose_train(data_name):
    if data_name == 'breast_cancer':
        dataset = load_breast_cancer()
        X = dataset.data
        Y = dataset.target

    elif data_name == 'heart':
        dataset = pd.read_csv('flask_MPC/static/datasets/heart/heart.csv')
        Y = dataset["target"]
        X = dataset.drop('target', axis=1)

    else:
        dataset = pd.read_csv('flask_MPC/static/datasets/NIMS/NIMS_Fatigue.csv')
        Y = dataset["Fatigue"]
        X = dataset.drop(['Fatigue', 'Tensile', 'Fracture', 'Hardness'], axis=1)
        Y.columns = ['Fatigue']
        for i in range(len(Y)):
            if Y[i] <= 505:
                Y.at[i] = 0
            else:
                Y.at[i] = 1

    return X,Y


def raw_score(model,x):
    """Compute the score of `x` by multiplying with the encrypted model,
    which is a vector of `paillier.EncryptedNumber`"""
    coef = model.coef_[0,:]
    score = model.intercept_[0]
    # print(x.nonzero())
    # print(self.weights)
    score += np.sum(np.multiply(x,coef))
    return score


def raw_evaluate(model,X):
    return [raw_score(model,X[i, :]) for i in range(X.shape[0])]

@app.route('/AI/score_result',methods=['POST'])
def calculate_score():
    enc_name = request.form.get('radio1')
    data_name = request.form.get('radio2')
    model_name = request.form.get('radio3')
    if data_name == 'NIMS':
        tag=['unqualified','qualified']
    else:
        tag=['ill','healthy']
    X,Y = choose_train(data_name)
    degree = find_next_power(X.shape[1])
    print(enc_name)
    if enc_name == 'CKKS':
        alice = CKKS_Alice(degree=degree)
        bob = CKKS_Bob(degree=degree,public_key=alice.public_key)
    else:
        alice = Phe_Alice()
        bob = Phe_Bob(alice.pubkey)
    rs = ShuffleSplit(n_splits=10,random_state=42,test_size=0.2)
    total_acc = []
    total_acc_diff=[]
    total_time=[]
    max_score_diff=[]
    total_ms_error=[]
    total_msl_error = []
    total_R_error=[]
    #x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #scaler = StandardScaler()
    #x_train = scaler.fit_transform(x_train)
    #x_test = scaler.transform(x_test)
    loc_path = 'flask_MPC/static/result_images/' + data_name + '/train_' + model_name + '.m'
    model = load(loc_path)
    r=1
    for train_ids, test_ids in rs.split(X, Y):
        print('===round ',r,'===')
        r += 1
        if type(X) == np.ndarray:
            x_train, x_test = [X[i] for i in train_ids], [X[i] for i in test_ids]
            y_train, y_test = [Y[i] for i in train_ids], [Y[i] for i in test_ids]
        else:
            x_train, x_test = [X.loc[i] for i in train_ids], [X.loc[i] for i in test_ids]
            y_train, y_test = [Y.loc[i] for i in train_ids], [Y.loc[i] for i in test_ids]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        time0 = time.perf_counter()
        encrypted_weights, encrypted_intercept = alice.encrypt_weights(model)
        bob.set_weights(encrypted_weights, encrypted_intercept)
        encrypted_scores = bob.encrypted_evaluate(x_test)
        scores = alice.decrypt_scores(encrypted_scores)
        raw = raw_evaluate(model, x_test)
        #print(scores)
        #print(raw)
        diff_score = np.max(abs(np.array(raw) - scores))
        max_score_diff.append(diff_score)
        for i in range(len(scores)):
            if scores[i] > 0:
                scores[i] = 1
            else:
                scores[i] = 0
        accuracy = (1 - np.mean(np.sign(scores) != y_test)) * 100
        total_acc.append(round(accuracy, 4))
        diff_accuracy = abs(model.score(x_test, y_test) * 100 - (1 - np.mean(np.sign(scores) != y_test)) * 100)
        total_acc_diff.append(round(diff_accuracy,3))
        time_cost_test = time.perf_counter() - time0
        total_time.append(round(time_cost_test,3))
        # 均方误差
        ms_error = mean_squared_error(y_test, scores)
        total_ms_error.append(round(ms_error,3))
        # 对数均方误差
        msl_error = mean_squared_log_error(y_test, scores)
        total_msl_error.append(round(msl_error,3))
        # R方
        R_error = r2_score(y_test, scores)
        total_R_error.append(round(R_error,3))
        conf_matrix = confusion_matrix(y_test, scores)
        draw_conf_matrix(conf_matrix, tag,
                         'flask_MPC/static/result_images/' + data_name + '/' + model_name + '_conf_matrix_' + enc_name + '.svg',
                         title='confusion matrix for ' + data_name + ' dataset with ' + enc_name + ' ' + model_name + ' model')
        false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, scores)
        draw_ROC_curve(model_name,
                       'ROC Curve for ' + data_name + ' dataset with ' + enc_name + ' ' + model_name + ' model',
                       false_positive_rate, true_positive_rate,
                       'flask_MPC/static/result_images/' + data_name + '/' + model_name + '_ROC_curve_' + enc_name + '.svg')
        print(accuracy)
        print(diff_accuracy)
        print(diff_score)
        print(time_cost_test)


    return render_template('AI_TEST_BACK_SCORE.html',
                           accuracy=round(np.mean(total_acc),3),
                           diff_accuracy=max(total_acc_diff),
                           time_cost_test = round(np.mean(total_time),3),
                           diff_score=max(max_score_diff),
                           MSE=round(np.mean(total_ms_error),3),
                           MSLE=round(np.mean(total_msl_error),3),
                           R_error=round(np.mean(total_R_error),3),
                           conf='../static/result_images/' + data_name + '/' + model_name + '_conf_matrix_'+enc_name+'.svg',
                           ROC='../static/result_images/' + data_name + '/' + model_name + '_ROC_curve_'+enc_name+'.svg')

