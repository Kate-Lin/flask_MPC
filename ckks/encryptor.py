from util.random_sample import sample_triangle
from util.polynomial import Poly
from util.ciphertext import Ciphertext

class CKKSEncryptor:
    def __init__(self, param, public_key, secret_key = None):
        self.public_key = public_key
        self.secret_key = secret_key
        self.poly_degree = param.poly_degree
        self.coeff_modulus = param.ciph_modulus
        self.big_modulus = param.big_modulus
        self.crt_context = param.crt_context

    def encrypt(self, plain):
        p0 = self.public_key[0]
        p1 = self.public_key[1]
        random_vec = Poly(self.poly_degree,sample_triangle(self.poly_degree))
        error1 = Poly(self.poly_degree, sample_triangle(self.poly_degree))
        error2 = Poly(self.poly_degree, sample_triangle(self.poly_degree))
        c0 = p0.multiply(random_vec,self.coeff_modulus,crt=self.crt_context)
        c0 = error1.add(c0,self.coeff_modulus)
        c0 = c0.add(plain.poly,self.coeff_modulus)
        c0 = c0.mod_small(self.coeff_modulus)

        c1 = p1.multiply(random_vec,self.coeff_modulus,crt=self.crt_context)
        c1 = error2.add(c1, self.coeff_modulus)
        c1 = c1.mod_small(self.coeff_modulus)

        return Ciphertext(c0,c1,plain.scaling_factor, self.coeff_modulus)