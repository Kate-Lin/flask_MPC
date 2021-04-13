from math import sqrt
import math

from util.polynomial import Poly
from util.ciphertext import Ciphertext
from util.plaintext import Plaintext

class CKKSEvaluator:
    def __init__(self,param = None, poly_degree = None):
        if param:
            self.degree = param.poly_degree
            self.crt_context = param.crt_context
        else:
            self.degree = poly_degree
            self.crt_context = None

    def add(self, cipher1, cipher2):
        assert isinstance(cipher1, Ciphertext)
        assert isinstance(cipher2, Ciphertext)
        assert cipher1.scaling_factor == cipher2.scaling_factor, "Scaling factors are not equal. " \
            + "Ciphertext 1 scaling factor: %d bits, Ciphertext 2 scaling factor: %d bits" \
            % (math.log(cipher1.scaling_factor, 2), math.log(cipher2.scaling_factor, 2))
        assert cipher1.modulus == cipher2.modulus, "Moduli are not equal. " \
            + "Ciphertext 1 modulus: %d bits, Ciphertext 2 modulus: %d bits" \
            % (math.log(cipher1.modulus, 2), math.log(cipher2.modulus, 2))

        modulus = cipher1.modulus
        c0 = cipher1.c0.add(cipher2.c0,modulus)
        c1 = cipher1.c1.add(cipher2.c1,modulus)
        return Ciphertext(c0,c1,cipher1.scaling_factor,cipher1.modulus)

    def add_plain(self,cipher,plain):
        assert isinstance(cipher,Ciphertext)
        assert isinstance(plain,Plaintext)
        assert cipher.scaling_factor == plain.scaling_factor, "Scaling factors are not equal. " \
                                    + "Ciphertext scaling factor: %d bits, Plaintext scaling factor: %d bits" \
                                    % (math.log(cipher.scaling_factor, 2),
                                    math.log(plain.scaling_factor, 2))
        c0 = cipher.c0.add(plain.poly,cipher.modulus)
        c0 = c0.mod_small(cipher.modulus)

        return Ciphertext(c0,cipher.c1,cipher.scaling_factor,cipher.modulus)

    def multiply_plain(self,cipher,plain):
        assert isinstance(cipher,Ciphertext)
        assert isinstance(plain,Plaintext)

        c0 = cipher.c0.multiply(plain.poly,cipher.modulus,crt=self.crt_context)
        c0 = c0.mod_small(cipher.modulus)
        c1 = cipher.c1.multiply(plain.poly,cipher.modulus,crt=self.crt_context)
        c1 = c1.mod_small(cipher.modulus)

        return Ciphertext(c0,c1,cipher.scaling_factor * plain.scaling_factor,cipher.modulus)

    def rescale(self, ciph, division_factor):
        """Rescales a ciphertext to a new scaling factor.

        Divides ciphertext by division factor, and updates scaling factor
        and ciphertext. modulus.

        Args:
            ciph (Ciphertext): Ciphertext to modify.
            division_factor (float): Factor to divide by.

        Returns:
            Rescaled ciphertext.
        """
        c0 = ciph.c0.scalar_integer_divide(division_factor)
        c1 = ciph.c1.scalar_integer_divide(division_factor)
        return Ciphertext(c0, c1, ciph.scaling_factor // division_factor,
                          ciph.modulus // division_factor)

    def lower_modulus(self, ciph, division_factor):
        """Rescales a ciphertext to a new scaling factor.

        Divides ciphertext by division factor, and updates scaling factor
        and ciphertext modulus.

        Args:
            ciph (Ciphertext): Ciphertext to modify.
            division_factor (float): Factor to divide by.

        Returns:
            Rescaled ciphertext.
        """
        new_modulus = ciph.modulus // division_factor
        c0 = ciph.c0.mod_small(new_modulus)
        c1 = ciph.c1.mod_small(new_modulus)
        return Ciphertext(c0, c1, ciph.scaling_factor, new_modulus)