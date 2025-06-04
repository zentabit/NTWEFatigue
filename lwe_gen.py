"""
Generate MLWE samples in the ring Z_q[x]/(x^n + 1).
Uses RingLWE from sage.crypto.lwe
"""
from sage.all import *
from sage.modules.free_module_element import vector
from numpy import array, zeros, identity, block,ones
# from scipy.linalg import circulant
from scipy.linalg import qr
from numpy.random import shuffle
from numpy import random
import numpy as np
# import matplotlib.pyplot as plt

from sage.crypto.lwe import RingLWE, samples
from sage.stats.distributions.discrete_gaussian_polynomial import DiscreteGaussianDistributionPolynomialSampler

def negacyclic(coeffs):
    """
    Given a vector of coefficients for a polynomial in the cyclotomic ring Z[X]/(X^n + 1),
    generates the negacyclic matrix representation.
    
    Args:
        coeffs (array-like): Coefficients of the polynomial (length n)
    
    Returns:
        numpy.ndarray: The n x n negacyclic matrix representation
    """
    coeffs = np.asarray(np.squeeze(coeffs))
    n = len(coeffs)
    
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    for i in range(n):
        row = np.roll(coeffs, i)
        if i > 0:
            row[:i] *= -1
        matrix[i] = row
    
    return matrix

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

def init_oracles(n, q, d = 1, sigma = 3.0):
    """
    Initiates d RLWE oracles according to settings.

    Args:
        n (int) : a power to 2, degree of polynomials
        d (int) : MLWE module rank (how many oracles)
        sigma (float) : standard error of Discrete Gaussian
        q (int) : prime, reduction modulus
    
    Returns:
        array[d, RingLWE] : a d-vector of RingLWE oracles
    """

    D = DiscreteGaussianDistributionPolynomialSampler(ZZ['x'], n=euler_phi(2*n), sigma=RealNumber(sigma))
    return [ RingLWE(N=2*n, q=q, D=D, secret_dist = 'noise') for _ in range(0,d) ]

def extract_secret(oracle):
    """
    Given a RLWE oracle from sage.crypto.lwe, extract its secret s

    Args:
        oracle (sage.crypto.lwe.RingLWE) : an oracle instance
    
    Returns:
        array[n] : an n-vector with the coefficient representation of s
    """
    s = getattr(oracle, '_RingLWE__s').coefficients(sparse = False)
    s = np.pad(s, (0,oracle.n - len(s)))
    return s

def mlwe_sample(oracles):
    """
    Provided a set of d RLWE oracles, draw an MLWE sample

    Args:
        oracles (array[d, sage.crypto.lwe.RingLWE]) : a d-vector of oracles

    Returns:
        array[(d,n)], array[(1,n)] : The pair a x (b = < a, s > + e) in R_q[X]^d x T
    """
    aa = vector([ o.R_q.random_element() for o in oracles ])
    ss = vector([ getattr(r, '_RingLWE__s') for r in oracles ])
    e = oracles[0].D()
    out = e + aa.dot_product(ss)
    # print(e.coefficients(sparse = False))

    return np.array([vector(a) for a in aa]), vector(out) # shape (d,n), (1,n)

def generate_instance(rs, m = 1):
    """
    Provided d RLWE oracles, construct m MLWE samples

    Args:
        rs (array[d, sage.crypto.lwe.RingLWE]) : a d-vector of oracles
        m (int) : The number of samples
    
    Returns:
        array[(m,d,n)], array[(m,n)], array[(d,n)] : The polynomials for MLWE all in coefficient representation
    """
    n = getattr(rs[0], 'n')
    q = getattr(rs[0], 'q')
    d = len(rs)
    a = np.zeros((m,d,n))
    b = np.zeros((m,n))

    for i in range(0,m):
        a[i, :, :], b[i, :] = mlwe_sample(rs)

    s = np.array([ extract_secret(r) for r in rs ]) # shape: (d, n)

    return a, b, s

def build_matrix(aa,bb,q):
    """
    Build the lattice basis for an MLWE instance

    Args:
        aa (array[(m,d,n)]) : MLWE
        bb (array[(m,n)]) : MLWE
        q (int, prime) : Modulus

    Returns:
        array[(n * (d + m + 1), n * (d + m + 1))], array[(mn, dn)], array[(mn, n)] : The lattice basis and the two blocks A and B
    """
    m,d,n = np.shape(aa)
    A = np.vstack([ np.hstack([ negacyclic(a) for a in a1]) for a1 in aa ]) # shape mn x dn
    B = np.vstack([ negacyclic(b) for b in bb ])
    
    mat = np.squeeze(block([[q * identity(m * n, dtype='long'), c_mod(-A,q), c_mod(B,q) ],
             [zeros((d * n,m * n), dtype='long'), identity(d * n, dtype='long'), zeros((d * n,n), dtype='long')],
             [zeros((n,m * n), dtype='long'),zeros((n,d * n), dtype='long'), identity(n, dtype='long')]]))

    return mat, A, B

def build_dense(aa,bb,ss,q):
    """
    Build the dense lattice basis for an MLWE instance

    Args:
        aa (array[(m,d,n)]) : MLWE
        bb (array[(m,n)]) : MLWE
        ss (array[(d,n)]) : MLWE
        q (int, prime) : Modulus

    Returns:
        array[(n * (d + m + 1), n)], array[(dn, n)] : Dense basis [ E, S, I_n ]^T and secret S
    """
    mat, _, _ = build_matrix(aa, bb, q)
    m,d,n = np.shape(aa)
    S = np.vstack([negacyclic(s) for s in ss])
    sec = block([[zeros((m * n, n), dtype='long')],
             [S],
             [identity(n, dtype='long')]])
    
    return c_mod(np.matmul(mat,sec), q), sec
    # return np.matmul(mat,sec), S

def init_mlwe_lattice(n, m, q, d = 1, sigma = 3.0):
    """
    Initialise MLWE and return lattice and dense basis
    """
    q = next_prime(Integer(q-1))

    rs = init_oracles(n,q,d,sigma)
    a,b,ss = generate_instance(rs, m)
    mat, _, _ = build_matrix(a,b,q)
    dense, _ = build_dense(a,b,ss,q)
 
    return mat, dense