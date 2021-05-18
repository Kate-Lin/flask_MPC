from ckks.encoder import CKKSEncoder
from ckks.evaluator import CKKSEvaluator

class CKKS_Bob:
    def __init__(self, degree, public_key):
        self.degree = degree
        self.public_key = public_key
        M = int(degree*4)
        poly_degree = M // 2
        self.encoder = CKKSEncoder(poly_degree=poly_degree)
        self.evaluator = CKKSEvaluator(poly_degree=poly_degree)

    def set_weights(self,coef,intercept):
        self.coef = coef
        self.intercept = intercept


    def encrypted_scores(self, x:list):
        """

        :param x(list): the test_x to calculate with encrypted weight
        :return: encrypted_scores(Ciphertext)
        """
        x = self.align_list(x)           #change x into list whose length is equal to degree     STILL RAW MESSAGE
        plain_x = self.encoder.encode(x,self.coef.scaling_factor)
        ciph = self.evaluator.multiply_plain(self.coef,plain_x)
        ciph = self.evaluator.rescale(ciph,self.coef.scaling_factor)
        self.intercept = self.evaluator.lower_modulus(self.intercept,self.intercept.modulus//ciph.modulus)
        score = self.evaluator.add(self.intercept, ciph)
        return score


    def encrypted_evaluate(self, X):
        return [self.encrypted_scores(X[i, :].tolist()) for i in range(X.shape[0])]      #shape_X (569, 30) shape_x: 30
        #return type:Ciphertext

    def align_list(self,x: list) -> list:
        length = len(x)
        zeros_num = int(self.degree - length)
        x.extend([0] * zeros_num)
        return x