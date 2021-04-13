from util.plaintext import Plaintext

class CKKSDecryptor:
    def __init__(self,param,secret_key):
        self.poly_degree = param.poly_degree
        self.crt_context = param.crt_context
        self.secret_key = secret_key

    def decrypt(self,ciphertext,c2=None):
        (c0,c1) = ciphertext.c0,ciphertext.c1
        message = c1.multiply(self.secret_key,ciphertext.modulus,crt=self.crt_context)
        message = c0.add(message,ciphertext.modulus)
        if c2:
            secret_key_squared = self.secret_key.multiply(self.secret_key,ciphertext.modulus)
            c2_message = c2.multiply(secret_key_squared, ciphertext.modulus, crt=self.crt_context)
            message = message.add(c2_message,ciphertext.modulus)
        message = message.mod_small(ciphertext.modulus)
        return Plaintext(message,ciphertext.scaling_factor)
