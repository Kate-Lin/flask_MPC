import time
from contextlib import contextmanager
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from ckks.encoder import CKKSEncoder
from ckks.keygen import CKKSKeyGenerator
from ckks.encryptor import CKKSEncryptor
from ckks.param import CKKSParameters
from ckks.decryptor import CKKSDecryptor
from ckks.evaluator import CKKSEvaluator


@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))


# the object who hold the model
class Alice:
    def __init__(self, degree, kernel = 'linear'):
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
        self.model = SVC(kernel=kernel)
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

    def decrypt_scores(self, encrypted_score:list):
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

if __name__ == '__main__':
    dataset = load_breast_cancer()
    X = dataset.data
    Y = dataset.target
    degree = find_next_power(X.shape[1])
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    alice = Alice(degree=degree)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=666)
    with timer() as t:
        alice.fit(train_x, train_y)
    with timer() as t:
        error = np.mean(alice.predict(test_x) != test_y)
    print("Error {:.3f}".format(error))
    alice.show(test_x, test_y)
    print("Alice: Encrypting classifier")
    with timer() as t:
        encrypted_weights, encrypted_intercept = alice.encrypt_weights()
    print("Bob: Scoring with encrypted classifier")
    bob = Bob(degree,alice.public_key)
    bob.set_weight(encrypted_weights,encrypted_intercept)
    with timer() as t:
        encrypted_scores = bob.encrypted_evaluate(test_x)
    print(type(encrypted_scores))       # type is list of Ciphertext
    print("Alice: Decrypting Bob's scores")
    with timer() as t:
        scores = alice.decrypt_scores(encrypted_scores)
    print(scores)
    for i in range(len(scores)):
        if scores[i] > 0:
            scores[i] = 1
        else:
            scores[i] = 0
    error = np.mean(np.sign(scores) != test_y)
    print("Error {:.3f}".format(error))
    print("准确率为：{:.3f}%".format((1 - error) * 100))