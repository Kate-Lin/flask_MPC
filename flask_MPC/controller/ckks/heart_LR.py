import time
from contextlib import contextmanager
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report,precision_recall_curve

from ckks.encoder import CKKSEncoder
from ckks.keygen import CKKSKeyGenerator
from ckks.encryptor import CKKSEncryptor
from ckks.param import CKKSParameters
from ckks.decryptor import CKKSDecryptor
from ckks.evaluator import CKKSEvaluator

import matplotlib.pyplot as plt
from util.conf_matrix import draw_conf_matrix

@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))


# the object who hold the model
class Alice:
    def __init__(self, degree):
        M = int(degree*4)
        poly_degree = M // 2
        scale = 1 << 28
        big_modulus = 1 << 1200
        ciph_modulus = 1 << 600
        self.param = CKKSParameters(
            poly_degree=poly_degree,
            ciph_modulus=ciph_modulus,
            big_modulus=big_modulus,
            scaling_factor=scale
        )
        self.model = LogisticRegression()
        self.keygen = CKKSKeyGenerator(self.param)
        self.public_key = self.keygen.public_key
        self.secret_key = self.keygen.secret_key
        self.encoder = CKKSEncoder(param=self.param)
        self.encryptor = CKKSEncryptor(self.param, self.public_key, self.secret_key)
        self.decryptor = CKKSDecryptor(self.param, self.secret_key)


    def fit(self, X, y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

    def show(self,x_test,y_test):
        print("准确率为：{:.3f}%".format( self.model.score(x_test, y_test)*100))

    def encrypt_weights(self):
        coef = self.model.coef_[0,:].tolist()   #as raw message
        coef = align_list(coef)
        plain_coef = self.encoder.encode(coef,self.param.scaling_factor)
        encrypted_coef = self.encryptor.encrypt(plain_coef)
        intercept = self.model.intercept_.tolist()
        intercept = align_list(intercept)
        plain_intercept = self.encoder.encode(intercept,self.param.scaling_factor)
        encrypted_intercept = self.encryptor.encrypt(plain_intercept)
        return encrypted_coef, encrypted_intercept

    def decrypt_scores(self, encrypted_scores:list):
        score = []
        for c in encrypted_scores:
            plain_ = self.decryptor.decrypt(c)
            message_ = np.real(self.encoder.decode(plain_))
            total = np.sum(message_)
            score.append(total)
        return score

class Bob:
    def __init__(self, degree, public_key):
        self.public_key = public_key
        M = int(degree*4)
        poly_degree = M // 2
        self.encoder = CKKSEncoder(poly_degree=poly_degree)
        self.evaluator = CKKSEvaluator(poly_degree=poly_degree)

    def set_weight(self,coef,intercept):
        self.coef = coef
        self.intercept = intercept


    def encrypted_scores(self, x:list):
        """

        :param x(list): the test_x to calculate with encrypted weight
        :return: encrypted_scores(Ciphertext)
        """
        x = align_list(x)           #change x into list whose length is equal to degree     STILL RAW MESSAGE
        plain_x = self.encoder.encode(x,self.coef.scaling_factor)
        ciph = self.evaluator.multiply_plain(self.coef,plain_x)
        ciph = self.evaluator.rescale(ciph,self.coef.scaling_factor)
        self.intercept = self.evaluator.lower_modulus(self.intercept,self.intercept.modulus//ciph.modulus)
        score = self.evaluator.add(self.intercept, ciph)
        return score


    def encrypted_evaluate(self, X):
        return [self.encrypted_scores(X[i, :].tolist()) for i in range(X.shape[0])]      #shape_X (569, 30) shape_x: 30
        #return type:Ciphertext



def find_next_power(x):
    return 2 ** math.ceil(math.log2(x))

def align_list(x:list) -> list:
    length = len(x)
    zeros_num = int(degree - length)
    x.extend([0] * zeros_num)
    return x

def get_conf_matrix(predicted_y,y,addr):
    conf_matrix = confusion_matrix(y,predicted_y)
    draw_conf_matrix(conf_matrix,['healthy','ill'],addr)
    plt.clf()
    plt.cla()

def get_PR_Curve(test_y,predicted_y,addr):
    precision, recall, _ = precision_recall_curve(test_y, predicted_y)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(addr,bbox_inches='tight')
    plt.clf()
    plt.cla()

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

def raw_scores(x,coef,inter):
    score = inter
    idx = x.nonzero()[0]
    for i in idx:
        score += x[i]*coef[i]
    return score

if __name__ == '__main__':
    dataset = pd.read_csv('flask_MPC/static/datasets/heart/heart.csv')
    Y = dataset["target"]
    X = dataset.drop('target', axis=1)
    degree = find_next_power(X.shape[1])
    print(degree)
    alice = Alice(degree=degree)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    with timer() as t:
        alice.fit(train_x, train_y)
    predicted_y = alice.predict(test_x)
    coef = alice.model.coef_[0,:]
    inter = alice.model.intercept_[0]
    score_raw2 = [raw_scores(test_x[i,:],coef,inter) for i in range(test_x.shape[0])]
    raw_false_positive_rate, raw_true_positive_rate, raw_threshold = roc_curve(test_y, predicted_y)
    with timer() as t:
        lr_acc_score = accuracy_score(test_y, predicted_y)
    print("Accuracy of Alice Logistic Regression:", lr_acc_score * 100, '\n')
    print("<--Saving the confusion matrix of the RAW LR model.-->")
    #get_conf_matrix(predicted_y,test_y,'result_images/heart_LR/Alice/confusion_matrix.jpg')
    print("<--Showing the classification report of the RAW LR model.-->")
    print(classification_report(test_y, predicted_y))
    print("<--Saving the PR Curve of the RAW LR model.-->")
    #get_PR_Curve(test_y,predicted_y,'result_images/heart_LR/Alice/PR_Curve.jpg')
    print("<--Saving ROC Curve of the RAW LR model.-->")
    #get_ROC_Curve(raw_false_positive_rate,raw_true_positive_rate,'result_images/heart_LR/Alice/ROC_Curve.jpg')
    print("Alice: Encrypting classifier")
    with timer() as t:
        encrypted_weights, encrypted_intercept = alice.encrypt_weights()
    print("Bob: Scoring with encrypted classifier")
    bob = Bob(degree,alice.public_key)
    bob.set_weight(encrypted_weights,encrypted_intercept)
    with timer() as t:
        encrypted_scores = bob.encrypted_evaluate(test_x)
    #print(type(encrypted_scores))       # type is list of Ciphertext
    print("Alice: Decrypting Bob's scores")
    with timer() as t:
        scores = alice.decrypt_scores(encrypted_scores)
    print(np.array(scores)-np.array(score_raw2))
    print("The max score diff between the raw and encrypted:",np.max(np.array(scores)-np.array(score_raw2)))
    print("The mean score diff between the raw and encrypted:", np.mean(np.array(scores) - np.array(score_raw2)))
    for i in range(len(scores)):
        if scores[i] > 0:
            scores[i] = 1
        else:
            scores[i] = 0
    #这里的score就等于predicted_y
    enc_false_positive_rate, enc_true_positive_rate, enc_threshold = roc_curve(test_y, scores)
    enc_acc_score = accuracy_score(test_y, scores)
    plt.cla()
    plt.clf()
    print("Accuracy of Bob testing Logistic Regression:", lr_acc_score * 100, '\n')
    print("<--Saving the confusion matrix of the ENCRYPTED LR model.-->")
    #get_conf_matrix(scores,test_y,'result_images/heart_LR/Bob/confusion_matrix.jpg')
    print("<--Showing the classification report of the RAW LR model.-->")
    print(classification_report(test_y, scores))
    print("<--Saving the PR Curve of the ENCRYPTED LR model.-->")
    #get_PR_Curve(test_y,scores,'result_images/heart_LR/Bob/PR_Curve.jpg')
    print("<--Saving ROC Curve of the ENCRYPTED LR model.-->")
    #get_ROC_Curve(enc_false_positive_rate,enc_true_positive_rate,'result_images/heart_LR/Bob/ROC_Curve.jpg')

