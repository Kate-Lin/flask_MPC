import numpy as np
from flask_MPC import app,db
from flask import request,render_template,flash,abort,url_for,redirect,session,Flask,g
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,\
    roc_curve,classification_report,mean_squared_error, mean_squared_log_error,r2_score
from sklearn.utils import shuffle
from joblib import dump,load
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
        dataset = pd.read_csv('flask_MPC/static/datasets/NIMS/NIMS_Fatigue.csv')
        Y = dataset["Fatigue"]
        X = dataset.drop(['Fatigue', 'Tensile', 'Fracture', 'Hardness'], axis=1)
        Y.columns = ['Fatigue']
        for i in range(len(Y)):
            if Y[i] <= 505:
                Y.at[i] = 0
            else:
                Y.at[i] = 1
    if os.path.exists('flask_MPC/static/result_images/' + data_name + '/train_' + model_name + '.m'):
        print('have existing '+model_name+' model')
        model = load('flask_MPC/static/result_images/' + data_name + '/train_' + model_name + '.m')
    else:
        print('doesn\'t have existing '+model_name+' model')
        if model_name == 'LR':
            model = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=5000)
        else:
            model = SVC(kernel='linear')
    return X,Y,model

def draw_ROC_curve(model_name,title,false_positive_rate, true_positive_rate,addr):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(false_positive_rate, true_positive_rate, label=model_name)
    plt.plot([0, 1], ls='--')
    plt.plot([0, 0], [1, 0], c='.5')
    plt.plot([1, 1], c='.5')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.savefig(addr)


def get_ROC_Curve(fp_rate,tp_rate,addr):
    #sns.set_style('whitegrid')
    plt.figure(figsize=(10, 5))
    plt.title('Reciver Operating Characterstic Curve')
    plt.plot(fp_rate, tp_rate, label='Logistic Regression')
    plt.plot([0, 1], ls='--')
    plt.plot([0, 0], [1, 0], c='.5')
    plt.plot([1, 1], c='.5')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.savefig(addr, bbox_inches='tight')
    plt.clf()
    plt.cla()

def get_conf_matrix(predicted_y,y,addr):
    conf_matrix = confusion_matrix(y,predicted_y)
    draw_conf_matrix(conf_matrix,['unqualified','qualified'],addr)
    plt.clf()
    plt.cla()

@app.route('/AI/train_result',methods=['POST'])
def train_model():
    data_name = request.form.get('radio1')
    model_name = request.form.get('radio2')
    X,Y,model = choose_train(data_name,model_name)
    tag=[]
    if data_name == 'NIMS':
        tag=['unqualified','qualified']
    else:
        tag=['ill','healthy']
    rs = ShuffleSplit(n_splits=10,random_state=42,test_size=0.2)
    total_score = []
    total_time=[]
    total_ms_error=[]
    total_msl_error = []
    total_R_error=[]
    classification={}
    for train_ids, test_ids in rs.split(X, Y):
        if type(X)==np.ndarray:
            x_train, x_test = [X[i] for i in train_ids], [X[i] for i in test_ids]
            y_train, y_test = [Y[i] for i in train_ids], [Y[i] for i in test_ids]
        else:
            x_train, x_test = [X.loc[i] for i in train_ids], [X.loc[i] for i in test_ids]
            y_train, y_test = [Y.loc[i] for i in train_ids], [Y.loc[i] for i in test_ids]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        time0 = time.perf_counter()
        model.fit(x_train,y_train)
        dump(model,'flask_MPC/static/result_images/'+data_name+'/train_'+model_name+'.m')
        model_predict = model.predict(x_test)
        conf_matrix = confusion_matrix(y_test,model_predict)
        classification = classification_report(y_test,model_predict,output_dict=True)
        #print(classification)
        time_cost_train = time.perf_counter()-time0
        total_time.append(round(time_cost_train,3))
        print(model.score(x_test, y_test) * 100)
        total_score.append(round(model.score(x_test, y_test) * 100,4))
        #均方误差
        ms_error = mean_squared_error(y_test,model_predict)
        total_ms_error.append(round(ms_error,4))
        #对数均方误差
        msl_error = mean_squared_log_error(y_test,model_predict)
        total_msl_error.append(round(msl_error,4))
        #R方
        R_error = r2_score(y_test,model_predict)
        total_R_error.append(round(R_error,4))
        print(ms_error)
        print(msl_error)
        print(R_error)
        #print(model_predict)
        #print(y_test)
        draw_conf_matrix(conf_matrix,tag,
                         'flask_MPC/static/result_images/'+data_name+'/'+model_name+'_conf_matrix_RAW.jpg',title='confusion matrix for '+ data_name+' dataset with '+model_name+' model')
        false_positive_rate, true_positive_rate, threshold = roc_curve(y_test,model_predict)
        draw_ROC_curve(model_name,'ROC Curve for '+ data_name+' dataset with '+model_name+' model',false_positive_rate,true_positive_rate,'flask_MPC/static/result_images/'+data_name+'/'+model_name+'_ROC_curve_RAW.jpg')

    return render_template('AI_TRAIN_SCORE.html',
                           accuracy=round(np.mean(total_score),4),
                           time_cost=round(np.mean(total_time),4),
                           MSE=round(np.mean(total_ms_error),4),
                           MSLE=round(np.mean(total_msl_error),4),
                           R_error = round(np.mean(total_R_error),4),
                           class_report = classification,
                           conf='../static/result_images/'+data_name+'/'+model_name+'_conf_matrix_RAW.jpg',
                           ROC='../static/result_images/'+data_name+'/'+model_name+'_ROC_curve_RAW.jpg'
                           )
