#-*- coding: utf-8 -*-
"""
File: lognormal.py
Author: Eddie Kelly
Date: 2024

This file is part of the Quantum algorithm for linear systems of equations for the multi-dimensional Black-Scholes equations project which was completed as part 
of the thesis https://mural.maynoothuniversity.ie/id/eprint/19288/.

License: MIT License
"""
import numpy as np
from scipy.stats import lognorm

def lognormal_1D_stock(S : float, S_0 : float, r : float, vol : float, t : float) -> float:
    '''
    Returns the probability density function of  the 
    price of a stock at time t assuming its initial
    value at time 0 is S_0. The stock price is assumed
    to follow geometric Brownian motion, corresponding 
    to the lognormal distribution. The actual mean 
    and variance are calculated from the parameters 
    characeterising the option.

    Inputs:
    -------
    
    S   : (float)   Stock price at time t 
    S_0 : (float)   Initial stock price
    r   : (float)   Risk-free rate
    vol : (float)   Volatility
    t   : (float)   Time

    Outputs:
    --------

    (float) : the probability density function of the stock price at time t
              given its initial value at time 0 is equal to S_0 
    '''
    mu = np.log(S_0)+(r-0.5*(vol**2))*t
    sigma = vol*np.sqrt(t)
    return lognorm.pdf(S,s=sigma,scale=np.exp(mu))

def lognormal_nD_stock(S : np.array, S_0 : np.array, r : float, cov : np.array, t : float) -> float:
    '''
    Returns the probability density function of  the 
    price of a stock at time t assuming its initial
    value at time 0 is S_0. The stock price is assumed
    to follow geometric Brownian motion, corresponding 
    to the lognormal distribution. The actual mean 
    and variance are calculated from the parameters 
    characeterising the option.

    Inputs:
    -------

    S    : (np.array)  Array of underlying stock prices at time t 
    S_0  : (np.array)  Array of initial stock prices at time 0
    r    : (float)     Risk-free rate
    cov  : (np.array)  Covariance matrix of the underlying stocks
    t    : (float)     Time

    Outputs:
    --------

    (float) : The joint probability density function the stock prices at time t
              equal to S given their initial values at time 0 are equal to the
              array S_0
    '''
    mu = np.log(S_0)+(r-0.5*np.diag(cov))*t
    sigma = np.sqrt(np.diag(cov)*t)
    d = len(S)
    return (1/(np.prod(S)*np.sqrt(np.linalg.det(cov)*(2*np.pi)**d)))*np.exp(-0.5*np.dot(np.log(S)-mu,np.dot(np.linalg.inv(cov),np.log(S)-mu)))


def normal_1D_stock(x : float, x_0 : float, r : float, vol : float, T_1 : float) -> float:
    mu = x_0 + (r-0.5*(vol**2))*T_1
    sigma = vol*np.sqrt(T_1)
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)


def normal_2D_stock(x : np.array, x_0 : np.array, r : float, cov : np.array, T_1 : np.array) -> float:
    mu = x_0 + (r-0.5*np.diag(cov))*T_1
    corr = cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
    return (1/((2*np.pi*T_1)*np.prod(np.sqrt(np.diag(cov)))*np.sqrt(np.linalg.det(corr))))*np.exp(-0.5*np.dot(x-mu,np.dot(np.linalg.inv(cov),x-mu)))


