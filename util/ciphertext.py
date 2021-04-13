class Ciphertext:
    def __init__(self,c0,c1,scaling_factor=None,modulus=None):
        self.c0 = c0
        self.c1 = c1
        self.scaling_factor = scaling_factor
        self.modulus = modulus

    def __str__(self):
        return 'c0: ' + str(self.c0) + '\n + c1: ' + str(self.c1)