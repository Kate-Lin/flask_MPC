import phe as paillier

class Phe_Alice:
    """
    Trains a Logistic Regression model on plaintext data,
    encrypts the model for remote use,
    decrypts encrypted scores using the paillier private key.
    """

    def __init__(self):
        self.pubkey, self.privkey = paillier.generate_paillier_keypair(n_length=3072)

    def encrypt_weights(self,model):
        coef = model.coef_[0, :]
        print(coef.shape)
        print(model.intercept_)
        encrypted_weights = [self.pubkey.encrypt(coef[i])
                             for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(model.intercept_[0])
        return encrypted_weights, encrypted_intercept

    def decrypt_scores(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]

