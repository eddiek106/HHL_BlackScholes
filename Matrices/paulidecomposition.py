# -*- coding: utf-8 -*-
"""
File: paulidecomposition.py
Author: Eddie Kelly
Date: 2024

This file is part of the Quantum algorithm for linear systems of equations for the multi-dimensional Black-Scholes equations project which was completed as part 
of the thesis https://mural.maynoothuniversity.ie/id/eprint/19288/.

License: MIT License
"""
import numpy as np
import matrixmod
from  matrixmod import triangular_block, triangular_block_modified
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

def is_power_of_two(n):
        return n & (n - 1) == 0 and n != 0

def singular_block(n):
    A = np.zeros((n,n))
    triangular_block(A,n-1,1,0)
    return A

def singular_block_modified(n):
    A = np.zeros((n,n))
    triangular_block_modified(A,n-1,1,0)
    return A


def pauli_decomposer(A):
    return SparsePauliOp.from_operator(A)

def num_elements_above_x(a,x):
    'Check how many elements in the array a are larger than x'
    count = np.sum(a > x)
    count += np.sum(a < -x)
    return count


def relative_strength(A,tol,p=False):
    if (is_power_of_two(len(A)) == False):
        raise ValueError("The input matrix is not a power of 2")
    
    else:
        B = pauli_decomposer(A)
        abslist=[]
        for i in range(0,len(B.paulis)):
            abslist.append(np.abs(B.coeffs[i]))
        if (p == 'Weights'):  
            print("--The relative weight of each term in Pauli--")
            for i in range(0,len(B.paulis)):
                print(B.paulis[i],abslist[i])      
            plt.plot(list(range(0,len(B.paulis))),abslist)
            plt.show()
        elif (p == 'MostSignificant'):
            d = num_elements_above_x(np.array(abslist),tol)
            print("There are",d,"s of Pauli which are above "
                "the given tolerance of ",tol,".")
        else:
            return num_elements_above_x(np.array(abslist),tol)


def rel_strength_plot(A,tol_range,p=False):
    ylist = []
    for i in tol_range:
        ylist.append(relative_strength(A,i,p))
    plt.plot(ylist,tol_range)
    plt.ylabel('Tolerance')
    plt.xlabel('Number of Pauli terms above tolerance')
    plt.title('Relative strength of Pauli terms')
    plt.show()

def N_largest_terms(A,N):
    B = pauli_decomposer(A)
    plist = []
    clist = []
    for i in range(0,len(B.paulis)):
        plist.append(str(B.paulis[i]))
        clist.append(np.absolute(B.coeffs[i]))
    zipped = zip(clist,plist)
    sortedlist = sorted(zipped)    
    sorted1, sorted2 = zip(*sortedlist)
    sorted1, sorted2 = sorted1[::-1],sorted2[::-1]
    for i in range(0,N):
        print("The",i+1,"th largest term is:",sorted2[i],"with a coefficient of",sorted1[i])


