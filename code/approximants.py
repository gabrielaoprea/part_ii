#!/usr/bin/env python

from pyscf import gto, scf
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.interpolate import pade
from numpy.polynomial import polynomial
from matplotlib import pyplot as plt

def get_c_matrix(f):
    n = len(f)
    c = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            c[i,i-j]=f[j]
    return c

def get_beg(n, l):
    i_mat = np.zeros((l, n+1))
    for i in range(n+1):
        i_mat[i,i] = 1
    return i_mat

def concat_3(c, c2, im, m,n,o):
    fin = np.zeros((m+n+o+2, m+n+o+3))
    for i in range(m+n+o+2):
        for j in range(m+1):
            fin[i,j] = im[i,j]
        for j in range(n+1):
            fin[i,j+m+1] = c[i,j]
        for j in range(o+1):
            fin[i,j+m+n+2] = c2[i,j]
    return fin

def quadratic_approximant(series, m, n, o):
    l = len(series)
    if(m+n+o+2>l):
        print('Cannot compute quadratic approximant to this order.')
        return 0 
    c = get_c_matrix(series)
    c_squared = c.dot(c)
    c_n = c[:m+n+o+2, :n+1]
    c_squared_n = c_squared[:m+n+o+2, :o+1]
    i_mat = get_beg(m, m+n+o+2)
    mat = concat_3(c_n, c_squared_n, i_mat,m,n,o)
    coeff = np.copy(mat[:,m+n+2])
    mat_f = np.delete(mat, m+n+2,1)
    coeff = np.array([coeff])
    coeff = -coeff.transpose()
    solution = np.linalg.solve(mat_f, coeff)
    r = solution[0:(m+1)]
    p = solution[m+1:(m+n+2)]
    q = solution[(m+n+2):(m+n+o+2)]
    r = r.transpose()
    r = r[0].tolist()
    p = p.transpose()
    p = p[0].tolist()
    q = q.transpose()
    q = q[0].tolist()
    q = [1] + q
    r_poly = polynomial.Polynomial(r)
    p_poly = polynomial.Polynomial(p)
    q_poly = polynomial.Polynomial(q)
    return r_poly, p_poly, q_poly

def get_beg_2(n):
    i_mat = np.zeros((2*n+1, n+1))
    for i in range(n+1):
        i_mat[i,i] = 1
    return i_mat

def concat_2(c, im, n):
    fin = np.zeros((2*n+1, 2*n+2))
    for i in range(2*n+1):
        for j in range(n+1):
            fin[i,j] = im[i,j]
            fin[i,j+n+1] = c[i,j]
    return fin

def linear_approximant(series, n):
    l = len(series)
    if(2*n+1>l):
        print('Cannot compute linear approximant to this order.')
        return 0 
    c = get_c_matrix(serie)
    c_n = c[:2*n+1,:n+1]
    i_mat = get_beg_2(n)
    mat = concat_2(c_n, i_mat, n)
    coeff = np.copy(mat[:,n+1])
    mat_f = np.delete(mat, n+1,1)
    coeff = np.array([coeff])
    coeff = -coeff.transpose
    solution = np.linalg.solve(mat_f, coeff)
    p = solution[0:(n+1)]
    q = solution[n+1:(2*n+1)]
    p = p.transpose()
    p = p[0].tolist()
    q = q.transpose()
    q = q[0].tolist()
    q = [1] + q
    p_poly = polynomial.Polynomial(p)
    q_poly = polynomial.Polynomial(q)
    return p_poly, q_poly


def get_values(r, p, q, value):
    plus = (-p(value) + np.sqrt(p(value)**2 - 4*q(value)*r(value)))/(2*q(value))
    minus = (-p(value) - np.sqrt(p(value)**2 - 4*q(value)*r(value)))/(2*q(value))
    return plus, minus