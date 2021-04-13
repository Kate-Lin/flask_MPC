import fractions
import math
import sys


class EncodedNumber(object):
    """
    The operations of addition and multiplication [1]_ must be
        preserved under this encoding. Namely:

        1. Decode(Encode(a) + Encode(b)) = a + b
        2. Decode(Encode(a) * Encode(b)) = a * b
    """
    BASE = 16
    LOG2_BASE = math.log(BASE,2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    def __init__(self, pubkey, encoding, exponent):
        self.pubkey = pubkey
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, pubkey, scalar, precision=None, max_exponent=None):
        """
        :param pubkey: public key
        :param scalar: the int or float to be encrypted
        :param precision:
            if `scalar`is a float, then this is set so that minimal precision is
            lost. Lower precision leads to smaller encodings, which
            might yield faster computation.
        :param max_exponent:
        Ensure that the exponent of the returned
            `EncryptedNumber` is at most this.
        :return:
            EncodedNumber: Encoded form of *scalar*, ready for encryption
            against *public_key*.

        We take the convention that a
        number x < n/3 is positive, and that a number x > 2n/3 is
        negative. The range n/3 < x < 2n/3 allows for overflow
        detection.
        """

        # calculate the maximum exponent for desired precision
        if precision is None:
            if isinstance(scalar,int):
                prec_exponent = 0
            elif isinstance(scalar,float):
                # the base-2 exponent of the float
                bin_exponent = math.frexp(scalar)[1]
                bin_significant = bin_exponent-cls.FLOAT_MANTISSA_BITS
                # corresponding base BASE exponent
                prec_exponent = math.floor(bin_significant/cls.LOG2_BASE)
            else:
                raise TypeError("Don't know the precision of type %s." % type(scalar))
        else:
            prec_exponent = math.floor(math.log(precision,cls.BASE))

        if max_exponent is None:
            exponent = prec_exponent
        else:
            exponent = min(max_exponent,prec_exponent)

        # Fraction 转化为分数形式
        int_rep = round(fractions.Fraction(scalar) * fractions.Fraction(cls.BASE)** -exponent)
        if abs(int_rep) > pubkey.max_int:
            raise ValueError("Integer should be within +/- %d but got %d." % (pubkey.n,int_rep))

        return cls(pubkey, int_rep % pubkey.n, exponent)

    def decode(self):
        """
        :return: the decoded number
        """
        # self.encoding: The encoded number to store. Must be positive and
        #         less than :attr:`~PaillierPublicKey.max_int`.
        if self.encoding >= self.pubkey.n:
            raise ValueError('Attempted to decode corrupted number.')
        elif self.encoding <= self.pubkey.max_int:
            mantissa = self.encoding
        elif self.encoding >= self.pubkey.n - self.pubkey.max_int:
            mantissa = self.encoding - self.pubkey.n
        else:
            raise OverflowError('Overflow detected in decrypted number')

        if self.exponent >= 0:
            return mantissa * self.BASE ** self.exponent
        else:
            try:
                return mantissa/self.BASE ** -self.exponent
            except OverflowError as e:
                raise OverflowError("decoded result too large for a float") from e

    def decrease_exponent(self,expo):
        """
        :param expo: the desired exponent
        :return: Encodednumber with the same value and the desired exponent
        """
        if expo > self.exponent:
            raise ValueError('New exponent %i should be more negative than'
                             'old exponent %i' % (expo, self.exponent))
        factor = pow(self.BASE, self.exponent-expo)
        new_encoding = self.encoding * factor % self.pubkey.n
        return self.__class__(self.pubkey, new_encoding, expo)
