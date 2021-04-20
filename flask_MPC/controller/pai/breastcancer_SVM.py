import time
from contextlib import contextmanager
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import phe as paillier

@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))

class Alice:
    """
    Trains a Logistic Regression model on plaintext data,
    encrypts the model for remote use,
    decrypts encrypted scores using the paillier private key.
    """

    def __init__(self,kernel='linear'):
        self.model = SVC(kernel=kernel)

    def generate_paillier_keypair(self, n_length):
        self.pubkey, self.privkey = paillier.generate_paillier_keypair(n_length=n_length)

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def show(self,x_test,y_test):
        print("准确率为：{:.3f}%".format( self.model.score(x_test, y_test)*100))

    def encrypt_weights(self):
        coef = self.model.coef_[0, :]
        encrypted_weights = [self.pubkey.encrypt(coef[i])
                             for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
        return encrypted_weights, encrypted_intercept

    def decrypt_scores(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]


class Bob:
    """
    Is given the encrypted model and the public key.

    Scores local plaintext data with the encrypted model, but cannot decrypt
    the scores without the private key held by Alice.
    """

    def __init__(self, pubkey):
        self.pubkey = pubkey

    def set_weights(self, weights, intercept):
        self.weights = weights
        self.intercept = intercept

    def encrypted_score(self, x):
        """Compute the score of `x` by multiplying with the encrypted model,
        which is a vector of `paillier.EncryptedNumber`"""
        score = self.intercept
        #print(x.nonzero())
        #print(self.weights)
        idx = x.nonzero()[0]
        for i in idx:
            score += x[i] * self.weights[i]
        return score

    def encrypted_evaluate(self, X):
        return [self.encrypted_score(X[i, :]) for i in range(X.shape[0])]


def train_dataset():
    X = dataset.data
    Y = dataset.target
    # index = (Y==0)
    # Y[index] = -1
    # print(X.shape)
    # print(Y.shape)
    # print(dataset.DESCR)
    # print('特征名称：')
    # print(dataset.feature_names)
    # print('分类名称：')
    # print(dataset.target_names)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    print("Alice: Generating paillier keypair")
    alice = Alice()
    alice.generate_paillier_keypair(n_length=3072)
    print("Alice: Learning breast_cancer cancer classifier")
    with timer() as t:
        alice.fit(x_train, y_train)
    print("Classify with model in the clear -- "
          "what Alice would get having Bob's data locally")
    with timer() as t:
        error = np.mean(alice.predict(x_test) != y_test)
    print("Error {:.3f}".format(error))
    alice.show(x_test, y_test)

    print("Alice: Encrypting classifier")
    with timer() as t:
        encrypted_weights, encrypted_intercept = alice.encrypt_weights()

    print("Bob: Scoring with encrypted classifier")
    bob = Bob(alice.pubkey)
    bob.set_weights(encrypted_weights, encrypted_intercept)

    with timer() as t:
        encrypted_scores = bob.encrypted_evaluate(x_test)

    print("Alice: Decrypting Bob's scores")
    with timer() as t:
        scores = alice.decrypt_scores(encrypted_scores)
    for i in range(len(scores)):
        if scores[i] > 0:
            scores[i] = 1
        else:
            scores[i] = 0
    error = np.mean(np.sign(scores) != y_test)
    print("Error {:.3f}".format(error))
    print("准确率为：{:.3f}%".format((1 - error) * 100))
    # print(scores)
    # print(y_test)


if __name__ == '__main__':
    dataset = load_breast_cancer()
    train_dataset()

