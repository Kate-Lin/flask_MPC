import numpy as np
from flask_MPC import app,db
from flask import request,render_template,flash,abort,url_for,redirect,session,Flask,g
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,\
    roc_curve,classification_report,mean_squared_error, mean_squared_log_error,r2_score
from util.conf_matrix import draw_conf_matrix
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

def choose_train(data_name,model_name):
    if data_name == 'breast_cancer':
        dataset = load_breast_cancer()
        X = dataset.data
        Y = dataset.target

    elif data_name == 'heart':
        dataset = pd.read_csv('flask_MPC/static/datasets/heart/heart.csv')
        Y = dataset["target"]
        X = dataset.drop('target', axis=1)

    else:
        dataset = pd.read_csv('flask_MPC/static/datasets/transfusion/transfusion.data')
        dataset.columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'Class']
        X = dataset.drop(columns='Class')
        Y = dataset['Class']
    if model_name == 'LR':
        model = LogisticRegression(multi_class='ovr',solver='liblinear')
    else:
        model = SVC(kernel='linear')

    return X,Y,model

def draw_ROC_curve(model_name,false_positive_rate, true_positive_rate,addr):
    plt.figure(figsize=(10, 5))
    plt.title('Reciver Operating Characterstic Curve')
    plt.plot(false_positive_rate, true_positive_rate, label=model_name)
    plt.plot([0, 1], ls='--')
    plt.plot([0, 0], [1, 0], c='.5')
    plt.plot([1, 1], c='.5')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.savefig(addr)


@app.route('/AI/train_result',methods=['POST'])
def train_model():
    data_name = request.form.get('radio1')
    model_name = request.form.get('radio2')
    X,Y,model = choose_train(data_name,model_name)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    time0 = time.perf_counter()
    model.fit(x_train,y_train)
    model_predict = model.predict(x_test)
    conf_matrix = confusion_matrix(y_test,model_predict)
    classification = classification_report(y_test,model_predict,output_dict=True)
    print(classification)
    time_cost_train = time.perf_counter()-time0
    print(model.score(x_test, y_test) * 100)
    #均方误差
    ms_error = mean_squared_error(y_test,model_predict)
    #对数均方误差
    msl_error = mean_squared_log_error(y_test,model_predict)
    #R方
    R_error = r2_score(y_test,model_predict)
    print(ms_error)
    print(msl_error)
    print(R_error)
    draw_conf_matrix(conf_matrix,['true','false'],
                     'flask_MPC/static/result_images/'+data_name+'/'+model_name+'_conf_matrix_RAW.jpg',title='confusion matrix for '+ data_name+' dataset')
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test,model_predict)
    draw_ROC_curve(model_name,false_positive_rate,true_positive_rate,'flask_MPC/static/result_images/'+data_name+'/'+model_name+'_ROC_curve_RAW.jpg')
    return render_template('AI_TRAIN_SCORE.html',
                           accuracy=round(model.score(x_test, y_test) * 100,3),
                           time_cost=round(time_cost_train,3),
                           MSE=round(ms_error,3),
                           MSLE=round(msl_error,3),
                           R_error = round(R_error,3),
                           class_report = classification,
                           conf='../static/result_images/'+data_name+'/'+model_name+'_conf_matrix_RAW.jpg',
                           ROC='../static/result_images/'+data_name+'/'+model_name+'_ROC_curve_RAW.jpg'
                           )
