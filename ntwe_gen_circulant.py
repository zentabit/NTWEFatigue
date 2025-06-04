"""
Generate one sample of NTWE in d = 1 and the ring Z_q[x]/(x^n - 1)
"""
from sage.all import *
from sage.modules.free_module_element import vector
from numpy import array, zeros, identity, block,ones
import numpy as np
import random

from sage.stats.distributions.discrete_gaussian_polynomial import DiscreteGaussianDistributionPolynomialSampler
np.set_printoptions(linewidth=200) # so we can look at the matrices

def circulant(coeffs):
    """
    Construct a matrix from the input vector, where row i is the vector shifted to the right by i steps

    Args:
        coeffs (array-like) : input vector of length n
    
    Returns:
        numpy.ndarray : n x n matrix with vector in circulant representation
    """
    coeffs = np.asarray(np.squeeze(coeffs))
    n = len(coeffs)
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    for i in range(n):
        matrix[i] = np.roll(coeffs, i)
    return matrix

# Convert all entries of x to integers
vec_int = np.vectorize(lambda x : int(x))

def inv_circulant(a_coeffs, q):
    """
    Invert a polynomial in Z_q[x] / (x^n - 1) using Sage
    """
    n = len(a_coeffs)
    Zq = Zmod(q)
    R = PolynomialRing(Zq, 'x')
    x = R.gen()

    a = R(a_coeffs.tolist())

    S = R.quotient(x ** n - 1, 'xbar')
    # xbar = S.gen()
    a_bar = S(a)

    try:
        a_inv = a_bar**(-1) # uses NTT when n and q prime
    except ZeroDivisionError:
        raise ZeroDivisionError(f"Polynomial is not invertible in Z_{q}[x]/(x^{n} - 1)")
    
    return vec_int(np.array(a_inv.matrix())) # remove sagemath integers from matrix rep
    # return np.pad(a_inv.lift().coefficients(sparse = False), n - a_inv.degree())

def sample(sampler, n):
    """
    Use a polynomial sampler to draw coefficients
    """
    sample = sampler().coefficients(sparse = False)
    return np.pad(sample, (0, n - len(sample)))

def ntwe_sample_circulant(n, q, sigma):
    """
    Initialise NTWE (d = 1) and draw one sample
    """
    sampler = DiscreteGaussianDistributionPolynomialSampler(ZZ['x'], n = n, sigma=sigma)

    S = circulant(sample(sampler, n))  
    E = circulant(sample(sampler, n)) 
    A = circulant([ ZZ.random_element(-q//2, q//2, distribution = 'uniform') for _ in range(0,n) ]) 

    while True:
            f = sample(sampler, n)
            try:
                Finv = inv_circulant(f, q)
                break
            except ZeroDivisionError:
                # print("failed inverse")
                continue

    B = c_mod( (A @ S + E) @ Finv, q )
    return A, B, S, E, circulant(vec_int(f))

def c_mod(arr, q):
    """
    Reduce arr to the set of representatives [-q//2, q//2]

    Args:
        arr (numpy.ndarray) : to be reduced
        q (int) : the modulus

    Returns:
        numpy.ndarray : arr but with every element reduced to [-q//2, q//2]
    """
    return np.mod(arr + q//2, q) - q//2

def build_matrix(A,B,q):
    """
    Construct the NTWE matrix given A and B in circulant representation
    """
    n, _ = np.shape(A)

    mat = np.squeeze(block([[q * identity(n, dtype='long'), -A, B ],
             [zeros((n,n), dtype='long'), identity(n, dtype='long'), zeros((n,n), dtype='long')],
             [zeros((n,n), dtype='long'),zeros((n,n), dtype='long'), identity(n, dtype='long')]]))

    return mat

def build_dense(A, B, S, F, q):
    """
    Construct dense sublattice basis given A, B, S, F in circulant matrix representation
    """
    mat = build_matrix(A, B, q)
    n, _ = np.shape(A)
    sec = block([[zeros((n, n), dtype='long')],
             [S],
             [F]])
    
    return c_mod(np.matmul(mat,sec), q), sec

def init_ntwe_lattice(n, q, sigma = 3.0, seed = None):
    """
    Initialise NTWE and return lattice and dense basis when d = 1. 
    q and n are assumed to be prime.
    """
    random.seed(int(seed)) # for python
    set_random_seed(int(seed)) # for sage

    A, B, S, E, F = ntwe_sample_circulant(n, q, sigma=sigma)
    mat = build_matrix(A, B, q)
    dense, _ = build_dense(A, B, S, F, q)
 
    return mat, dense, E, S, F
