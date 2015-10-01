from numpy import zeros, linspace, atleast_2d, dot, ones, eye, nan
import numpy.random as rng
from numpy.random import RandomState
rng = RandomState()
from numpy cimport ndarray
from timeit import timeit
from numpy.linalg import cholesky, pinv
from sys import maxint
import pdb


cdef extern from "math.h":
    double pow(double a, double b)
    double exp(double a)
    double sqrt(double a)


cpdef ndarray[double, ndim=2] K_se(ndarray[double, ndim=1] Xi,\
                                   ndarray[double, ndim=1] Xj,\
                                   double lenscale,
                                   double sigvar):
    """
    cpdef ndarray[double, ndim=2] K_se(ndarray[double, ndim=1] Xi,\
                                   ndarray[double, ndim=1] Xj,\
                                   double lenscale,
                                   double sigvar):
    get a covariance matrix K using squared exponential kernel with
    lengthscale lenscale and signal variance sigvar for all pairs of
    input-space points in Xi and Xj, WHICH ARE BOTH 1D"""

    cdef int Ni = len(Xi)
    cdef int Nj = len(Xj)
    cdef ndarray[double, ndim=2] K = zeros([Ni, Nj])
    cdef int i, j
    for i in xrange(Ni):
        for j in xrange(Nj):
            K[i,j] = pow(sigvar, 2.) *\
                     exp(-(1./(2.*pow(lenscale,2.))) *\
                         pow(Xi[i] - Xj[j], 2.))
    return K


cpdef ndarray[double, ndim=1] sample(ndarray[double, ndim=1] X,
                                     ndarray[double, ndim=1] mu,
                                     ndarray[double, ndim=2] covmat,
                                     double noisevar2):
    """cpdef ndarray[double, ndim=1] sample(ndarray[double, ndim=1] X,
                                     ndarray[double, ndim=1] mu,
                                     ndarray[double, ndim=2] covmat,
                                     double noisevar2)"""
    cdef int nI = X.shape[0]
    covmat += eye(nI) * noisevar2
    covmat += eye(nI) * noisevar2
    cdef ndarray[double, ndim=2] Lun = cholesky(covmat)
    # draw samples with our shiny new posterior!
    cdef ndarray[double, ndim=1] seeds = rng.randn(nI)
    cdef ndarray[double, ndim=1] sample = (mu + dot(Lun, seeds))
    return sample


cpdef ndarray[double, ndim=1] conditioned_mu(ndarray[double, ndim=1] X,
                                             ndarray[double, ndim=1] xObs,
                                             ndarray[double, ndim=1] yObs,
                                             double lenscale,
                                             double sigvar,
                                             double noisevar2):
    """cpdef ndarray[double, ndim=1] conditioned_mu(ndarray[double, ndim=1] X,
                                             ndarray[double, ndim=1] xObs,
                                             ndarray[double, ndim=1] yObs,
                                             double lenscale,
                                             double sigvar,
                                             double noisevar2)"""
    cdef int nI = X.shape[0]
    cdef int nJ = xObs.shape[0]

    cdef ndarray[double, ndim=2] k=\
            K_se(xObs, X, lenscale, sigvar)
    # get covarmat for observed points
    cdef ndarray[double, ndim=2] Kobs =\
        K_se(xObs, xObs, lenscale, sigvar) + eye(nJ) * noisevar2

    cdef ndarray[double, ndim=2] invKobs = pinv(Kobs)  # invert

    return dot(dot(k.T, invKobs), yObs)  # exp val for points given obs


cpdef ndarray[double, ndim=2] conditioned_covmat(ndarray[double, ndim=1] X,
                                                ndarray[double, ndim=2] KX,
                                                ndarray[double, ndim=1] xObs,
                                                double lenscale,
                                                double sigvar,
                                                double noisevar2):
    """cpdef ndarray[double, ndim=2] conditioned_covmat(ndarray[double, ndim=1] X,
                                                ndarray[double, ndim=2] KX,
                                                ndarray[double, ndim=1] xObs,
                                                double lenscale,
                                                double sigvar,
                                                double noisevar2)"""
    cdef int nI = X.shape[0]
    cdef int nJ = xObs.shape[0]

    cdef ndarray[double, ndim=2] k=\
            K_se(xObs, X, lenscale, sigvar)
    # get covarmat for observed points
    cdef ndarray[double, ndim=2] Kobs =\
            K_se(xObs, xObs, lenscale, sigvar) + eye(nJ) * noisevar2

    cdef ndarray[double, ndim=2] invKobs = pinv(Kobs)  # invert

    # certainty of vals | obs
    return KX - dot(dot(k.T, invKobs), k)

