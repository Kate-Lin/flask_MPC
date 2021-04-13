import numpy as np
from numpy.polynomial import Polynomial
from util.polynomial import Poly
from util.plaintext import Plaintext

class CKKSEncoder:
    """Basic CKKS encoder to encode complex vectors into polynomials."""
    def __init__(self,param = None, poly_degree = None):
        if param:
            self.M = param.poly_degree * 2
            self.degree = param.poly_degree
        else:
            self.M = poly_degree * 2
            self.degree = poly_degree
        self.xi = np.exp(2*np.pi*1j/self.M)
        self.create_sigma_R_basis()

    @staticmethod
    def vandermonde(xi:np.complex128, M:int) -> np.array:
        N = M //2
        matrix = []
        for i in range(N):
            root = xi ** (2 * i + 1)
            row = []
            for j in range(N):
                row.append(root**j)
            matrix.append(row)
        return matrix

    def sigma_inverse(self, b: np.array) -> Polynomial:
        """Encodes the vector b in a polynomial using an M-th root of unity."""
        A = CKKSEncoder.vandermonde(self.xi, self.M)
        coeffs = np.linalg.solve(A,b)
        p = Polynomial(coeffs)
        return p

    def sigma(self, p:Polynomial) -> np.array:
        """Decodes a polynomial by applying it to the M-th roots of unity."""
        outputs = []
        N = self.M // 2
        for i in range(N):
            root = self.xi**(2*i+1)
            output = p(root)
            outputs.append(output)
        return np.array(outputs)

    def pi(self,z: np.array) -> np.array:
        """Projects a vector of H into C^{N/2}."""
        N = self.M // 4
        return z[:N]

    def pi_inverse(self, z:np.array) -> np.array:
        """Expands a vector of C^{N/2} by expanding it with its
        complex conjugate."""
        z_conjugate = z[::-1]
        z_conjugate = [np.conjugate(x) for x in z_conjugate]
        return np.concatenate([z, z_conjugate])

    def create_sigma_R_basis(self):
        """Creates the basis (sigma(1), sigma(X), ..., sigma(X** N-1))."""
        self.sigma_R_basis = np.array(self.vandermonde(self.xi, self.M)).T

    def compute_basis_coordinates(self, z):
        output = np.array([np.real(np.vdot(z, b) / np.vdot(b,b)) for b in self.sigma_R_basis])
        return output

    def sigma_R_discretization(self, z):
        """Projects a vector on the lattice using coordinate wise random rounding."""
        coordinates = self.compute_basis_coordinates(z)

        rounded_coordinates = coordinate_wise_random_rounding(coordinates)
        y = np.matmul(self.sigma_R_basis.T, rounded_coordinates)
        return y


    def encode(self, z, scaling_factor: float) -> Plaintext:
        """Encodes a vector by expanding it first to H,
        scale it, project it on the lattice of sigma(R), and performs
        sigma inverse.
        """
        pi_z = self.pi_inverse(z)
        scaled_pi_z = scaling_factor * pi_z
        rounded_scale_pi_zi = self.sigma_R_discretization(scaled_pi_z)
        p = self.sigma_inverse(rounded_scale_pi_zi)

        # We round it afterwards due to numerical imprecision
        coef = np.round(np.real(p.coef)).astype(int).tolist()
        p = Poly(self.degree,coef)

        return Plaintext(p,scaling_factor=scaling_factor)

    def decode(self, p: Plaintext) -> np.array:
        """Decodes a polynomial by removing the scale,
        evaluating on the roots, and project it on C^(N/2)"""
        scale = p.scaling_factor
        p = Polynomial(np.array(p.poly.coeffs))
        rescaled_p = p / scale
        z = self.sigma(rescaled_p)
        pi_z = self.pi(z)
        return pi_z


def round_coordinates(coordinates):
    """Gives the integral rest."""
    coordinates = coordinates - np.floor(coordinates)
    return coordinates


def coordinate_wise_random_rounding(coordinates):
    """Rounds coordinates randonmly."""
    r = round_coordinates(coordinates)
    f = np.array([np.random.choice([c, c - 1], 1, p=[1 - c, c]) for c in r]).reshape(-1)

    rounded_coordinates = coordinates - f
    rounded_coordinates = [int(coeff) for coeff in rounded_coordinates]
    return rounded_coordinates

