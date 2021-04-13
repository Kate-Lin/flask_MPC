import random
import numpy as np
from numpy.polynomial import Polynomial
from util.polynomial import Poly

class CKKSKeyGenerator:
    def __init__(self, param):
        self.degree = param.poly_degree
        self.scaling_factor = param.scaling_factor
        self.hamming = self.degree // 4
        self.generate_secret_key()
        self.generate_public_key()

    def generate_secret_key(self):
        sample = [0] * self.degree
        total_weight = 0
        while total_weight < self.hamming:
            index = random.randrange(0,self.degree)
            if sample[index] == 0:
                r = random.randrange(0,1)
                if r == 0:
                    sample[index] = -1
                else:
                    sample[index] = 1
                total_weight += 1
        self.secret_key = Poly(self.degree,sample)

    def generate_public_key(self):
        mod = 1 << 1200
        pk_coeff = Poly(self.degree,[random.randrange(0, mod) for _ in range(self.degree)])
        sample = [0] * self.degree
        for i in range(self.degree):
            r = random.randrange(0,4)
            if r == 0:
                sample[i] = -1
            elif r == 1:
                sample[i] = 1
            else:
                sample[i] = 0
        pk_error = Poly(self.degree,sample)
        p0 = pk_coeff.multiply(self.secret_key,mod)
        p0 = p0.scalar_multiply(-1, mod)
        p0 = p0.add(pk_error, mod)
        p1 = pk_coeff
        self.public_key = (p0,p1)
