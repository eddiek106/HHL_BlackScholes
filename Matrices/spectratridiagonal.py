# -*- coding: utf-8 -*-
"""
File: spectratridiagonal.py
Author: Eddie Kelly
Date: 2024

This file is part of the Quantum algorithm for linear systems of equations for the multi-dimensional Black-Scholes equations project which was completed as part 
of the thesis https://mural.maynoothuniversity.ie/id/eprint/19288/.

License: MIT License
"""

import numpy as np

def tridiagonal_toeplitz_1(n,h):
    '''
    Produces a numpy array with a tridiagonal, Toeplitz
    matrix with 1's on the superdiagonal and -1's on the
    subdiagonal. This will correspond to the central first 
    order central difference approximation of the first derivative.
    This corresponds to a spatial discretization with stepsize h 
    and will correspond to a matrix of size n x n.

    Inputs:
    ------- 

    n (int)   : Dimension of matrix
    h (float) : Absolute stepsize of spatial disretization

    Outputs:
    --------

    (numpy.ndarray) : Tridiagonal matrix of size n x n 

    '''
    A = np.zeros((n,n))
    for i in range(0,n-1):
        A[i+1,i] = -1
        A[i,i+1] = 1
    return A/(2*h)

def tridiagonal_toeplitz_2(n,h):
    '''
    Produces a numpy array with a tridiagonal, Toeplitz
    matrix with 1's  on the superdiagonal and subdiagonal and -2 on
    the main diagonal. This will correspond to the central second
    order central difference approximation of the second derivative
    . This corresponds to a spatial discretization with stepsize h
    and will correspond to a matrix of size n x n.

    Inputs:
    ------- 

    n : (int)  Dimension of matrix
    h : (float)  Absolute stepsize of spatial disretization

    Outputs:
    --------

    (numpy.ndarray) : tridiagonal matrix of size n x n 

    '''

    A = -2*np.eye(n)
    for i in range(0,n-1):
        A[i+1,i] = 1
        A[i,i+1] = 1
    return A/(h**2)

def tridiag(a,b,c,n):
    '''
    A tridiagonal Toeplitz matrix with values 'a','b','c' 
    on the sub, main, and super diagonals respectively.

    Inputs:
    ------- 
    
    a : (float)  Value of subdiagonal

    b : (float)  Value of main diagonal

    c : (float)  Value of superdiagonal

    n : (int)  Dimension of matrix

    Outputs:
    --------

    (numpy.ndarray) : tridiagonal matrix of size n x n
    '''
    A = np.zeros((n,n))
    np.fill_diagonal(A[1:],a)
    np.fill_diagonal(A,b)
    np.fill_diagonal(A[:,1:],c)
    return A


def tridiag_mod(a,b,c,n):
    '''
    A tridiagonal Toeplitz matrix with values 'a' on the second subdiagonal,
    'b' on the main diagonal, and 'c' on the second superdiagonal.

    Inputs:
    -------

    a (float) : value of second subdiagonal
    b (float) : value of main diagonal
    c (float) : value of second superdiagonal
    n (int) : dimension of matrix

    Outputs:
    --------

    A (numpy array) : tridiagonal matrix of size n x n
    
    '''
    A = np.zeros((n,n))
    np.fill_diagonal(A[2:],a)
    np.fill_diagonal(A,b)
    np.fill_diagonal(A[:,2:],c)
    return A
