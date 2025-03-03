#-*- coding: utf-8 -*-
"""
File: bs_differential_op.py
Author: Eddie Kelly
Date: 2024

This file is part of the Quantum algorithm for linear systems of equations for the multi-dimensional Black-Scholes equations project which was completed as part 
of the thesis https://mural.maynoothuniversity.ie/id/eprint/19288/.

License: MIT License
"""

import sys
sys.path.append("..")

import numpy as np
from scipy.stats import norm
from Matrices import spectratridiagonal as st, matrixmod as mm

def black_scholes_1D_prop(n : int, h : float, r : float, vol : float) -> np.ndarray:
    '''
    Outputs the matrix A corresponding to the discretization
    operator for the Black-Scholes PDE in one dimension..

    Inputs:
    ------- 

    n   : (int)  Number of grid points
    h   : (float)  Grid spacing
    r   : (float)  Risk-free rate
    vol : (float)  Volatility

    Outputs:
    --------

    (np.ndarray) : matrix A for the Black-Scholes PDE in one dimension
    '''

    return 0.5*(vol**2)*(st.tridiagonal_toeplitz_2(n,h))+(r-0.5*(vol**2))*st.tridiagonal_toeplitz_1(n,h)
    
def black_scholes_1D_improp(n : int, h : float, r : float, vol : float) -> np.ndarray:
    '''
    Outputs the matrix A for the Black-Scholes PDE in one dimension. However, in this instance
    the disrectization matrix for the central second difference is replaced by the the central
    first difference matrix squared.

    Inputs:
    -------

    n   : (int)  Number of grid points
    h   : (float)  Grid spacing
    r   : (float)  Risk-free rate
    vol : (float)  Volatility

    Outputs:
    --------

    (np.ndarray) : matrix A for the Black-Scholes PDE in one dimension
    '''

    return 0.5*(vol**2)*(np.dot(st.tridiagonal_toeplitz_1(n,h),st.tridiagonal_toeplitz_1(n,h)))+(r-0.5*(vol**2))*st.tridiagonal_toeplitz_1(n,h)    

def black_scholes_1D_analytic(S_0 : float, K : float, r : float,
                               vol : float, T : float, type : str) -> float:
    '''
    Returns the price of a European option using the Black-Scholes formula.

    Inputs:
    -------

    S_0 :  (float)  Initial stock price
    K   :  (float)  Strike price
    r   :  (float)  Risk-free rate
    vol :  (float)  Volatility
    T   :  (float)  Time to maturity
    type : (str)  Option type ('call' or 'put')

    Outputs:
    -------- 

    (float) : price of the option
    '''

    d_plus = (1/(vol*np.sqrt(T)))*(np.log(S_0/K)+(r + 0.5*(vol**2))*T)
    d_minus = d_plus - vol*np.sqrt(T)
    if (type == 'call'):
        return S_0*norm.cdf(d_plus)-K*np.exp(-r*T)*norm.cdf(d_minus)
    elif (type == 'put'):
        return K*np.exp(-r*T)*norm.cdf(-d_minus)-S_0*norm.cdf(-d_plus)
    else:
        raise ValueError('Invalid option type')

def kron_repeat(A : np.ndarray, n : int) -> np.ndarray:
    '''
    Ouputs the Kronecker product of a matrix A with itself n times.
    
    Inputs:
    -------

    A : (np.ndarray)  matrix to be repeated in the Kronecker product
    n : (int)  number of times matrix A is 'repeated' in Kronecker product

    Outputs:
    --------

    (np.ndarray) : Kronecker product of A with itself n times
    '''

    result = A
    for _ in range(0,n-1):
        result = np.kron(result,A)
    return result

def first_central_diff(n : int, h : float, pos : int, d : int) -> np.ndarray:
    '''
    Outputs the matrix for the central first difference operator in d-dimensions.
    
    Inputs:
    -------

    n   : (int)  number of grid points
    h   : (float)  grid spacing
    pos : (int)  axis with respect to which the central first difference operator is taken (1 <= pos <= d)
    d   : (int)  dimension of the problem

    Outputs:
    --------

    (np.ndarray) : matrix for the central first difference operator in d-dimensions
    '''

    if d == 1:
        return st.tridiagonal_toeplitz_1(n,h)
    if pos == d:
        return np.kron(st.tridiagonal_toeplitz_1(n,h),kron_repeat(np.eye(n),d-1))

    elif pos == 1:   
        return np.kron(kron_repeat(np.eye(n),d-1),st.tridiagonal_toeplitz_1(n,h)) 

    else: 
        return np.kron(np.kron(kron_repeat(np.eye(n),d-pos),st.tridiagonal_toeplitz_1(n,h)),kron_repeat(np.eye(n),pos-1))

def second_central_diff(n : int, h : float, pos : int, d : int) -> np.ndarray:
    '''
    Ouputs the matrix for the central second difference operator in d-dimensions.

    Inputs:
    -------

    n   : (int)  number of grid points
    h   : (float)  grid spacing
    pos : (int)  axis with respect to which the central first difference operator is taken (1 <= pos <= d)
    d   : (int)  dimension of the problem

    Outputs:
    --------

    (np.ndarray) : matrix for the central second difference operator in d-dimensions
    '''

    if d == 1:
        return st.tridiagonal_toeplitz_2(n,h)
    
    if pos == d:
        return np.kron(st.tridiagonal_toeplitz_2(n,h),kron_repeat(np.eye(n),d-1))
    
    elif pos == 1:   
        return np.kron(kron_repeat(np.eye(n),d-1),st.tridiagonal_toeplitz_2(n,h))
    
    else:
        return np.kron(np.kron(kron_repeat(np.eye(n),d-pos),st.tridiagonal_toeplitz_2(n,h)),kron_repeat(np.eye(n),pos-1))
    
def mixed_central_diff(n : int, h : float, pos_i : int, pos_j : int, d : int) -> np.ndarray:
    '''
    Outputs the matrix for the mixed central difference operator in d-dimensions.

    Inputs:
    -------

    n     : (int)   number of grid points
    h     : (float) grid spacing
    pos_i : (int)   axis with respect to which the first central difference operator is taken (1 <= pos_i <= d)
    pos_j : (int)   axis with respect to which the first central difference operator is taken (1 <= pos_j <= d)
    d     : (int)   dimension of the problem

    It is assumed t that pos_i < pos_j.

    Outputs:
    --------

    (np.ndarray) : matrix for the mixed central difference operator in d-dimensions
    '''
    if pos_i >= pos_j:
        raise ValueError('pos_j must be greater than pos_i')
    
    return (first_central_diff(n,h,pos_i,d)) @ (first_central_diff(n,h,pos_j,d))
              
def black_scholes_nD_prop(n : int, h : float, r : float, cov_m : np.ndarray, d : int) -> np.ndarray:
    '''
    Outputs the matrix A corresponding to the discretization
    operator for the Black-Scholes PDE in d dimension.

    Inputs:
    -------

    n      : (int)  number of grid points
    h      : (float)  grid spacing
    r      : (float)  risk-free rate
    cov_m  : (np.ndarray)  covariance matrix
    d      : (int) dimension of the problem
    
    Outputs:
    --------

    (np.ndarray) : matrix A for the Black-Scholes PDE in d dimensions
    '''
    A = np.zeros((n**d,n**d))
    for i in range(1,d+1):
        A += 0.5*(cov_m[i-1,i-1])*second_central_diff(n,h,i,d) 
    for i in range(1,d+1):
        for j in range(i+1,d+1):
                A += cov_m[i-1,j-1]*mixed_central_diff(n,h,i,j,d)
    for i in range(1,d+1):
        A += (r-0.5*(cov_m[i-1,i-1]))*first_central_diff(n,h,i,d)
    return A 

def black_scholes_nD_improp(n,h,r,cov_m,d):
    '''
    Outputs the matrix A corresponding to the discretization
    operator for the Black-Scholes PDE in d dimension. However, in this instance
    the disrectization matrix for the central second difference is replaced by the the central
    first difference matrix squared.

    Inputs:
    -------

    n      : (int)  number of grid points
    h      : (float)  grid spacing
    r      : (float)  risk-free rate
    cov_m  : (np.ndarray)  covariance matrix
    d      : (int) dimension of the problem

    Outputs:
    --------

    (np.ndarray) : matrix A for the Black-Scholes PDE in d dimensions
    
    '''
    A = np.zeros((n**d,n**d))
    for i in range(1,d+1):
        A += 0.5*(cov_m[i-1,i-1])*(first_central_diff(n,h,i,d) @ first_central_diff(n,h,i,d))
    for i in range(1,d+1):
        for j in range(i+1,d+1):
                A += cov_m[i-1,j-1]*mixed_central_diff(n,h,i,j,d)
    for i in range(1,d+1):
        A += (r-0.5*(cov_m[i-1,i-1]))*first_central_diff(n,h,i,d)
    return A 
