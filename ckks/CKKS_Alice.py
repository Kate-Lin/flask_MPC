from ckks.param import CKKSParameters
from ckks.keygen import CKKSKeyGenerator
from ckks.encryptor import CKKSEncryptor
from ckks.encoder import CKKSEncoder
from ckks.decryptor import CKKSDecryptor
import math
import numpy as np

# the object who hold the model
class CKKS_Alice:
    def __init__(self, degree):
        self.degree = degree
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
        self.keygen = CKKSKeyGenerator(self.param)
        self.public_key = self.keygen.public_key
        self.secret_key = self.keygen.secret_key
        self.encoder = CKKSEncoder(param=self.param)
        self.encryptor = CKKSEncryptor(self.param, self.public_key, self.secret_key)
        self.decryptor = CKKSDecryptor(self.param, self.secret_key)


    def encrypt_weights(self,model):
        coef = model.coef_[0,:].tolist()   #as raw message
        coef = self.align_list(coef)
        plain_coef = self.encoder.encode(coef,self.param.scaling_factor)
        encrypted_coef = self.encryptor.encrypt(plain_coef)
        intercept = model.intercept_.tolist()
        intercept = self.align_list(intercept)
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

    def align_list(self,x:list) -> list:
        length = len(x)
        zeros_num = int(self.degree - length)
        x.extend([0] * zeros_num)
        return x