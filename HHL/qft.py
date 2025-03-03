# -*- coding: utf-8 -*-
"""
File: qft.py
Author: Eddie Kelly
Date: 2024

This file is part of the Quantum algorithm for linear systems of equations for the multi-dimensional Black-Scholes equations project which was completed as part 
of the thesis https://mural.maynoothuniversity.ie/id/eprint/19288/.

License: MIT License
"""

import numpy as np
import qiskit
import matplotlib.pyplot as plt 

def qft(circuit : qiskit.QuantumCircuit ,conv : str, swap_before : bool, swap_after : bool, sign : int) -> qiskit.QuantumCircuit:
    '''
    For a given circuit 'circuit', it will apply
    quantum Fourier transform (QFT).

    Inputs:
    -------

    circuit : (qiskit.QuantumCircuit)  a quantum circuit

    conv : (str)  'little-endian' or 'big-endian' convention for QFT

    swap_before : (bool)  whether to apply swap gates before QFT

    swap_after : (bool)  whether to apply swap gates after QFT

    sign : (int)  1 or -1, sign of the phase gate.

    Outputs:
    --------

    (qiskit.QuantumCircuit) The previous quantum circuit with the QFT applied to it.                

    '''
    n = circuit.num_qubits
    #circuit.barrier()
    if swap_before:
        for k in range(n//2):
            circuit.swap(k, n-1-k)
        #circuit.barrier()
    else:
        pass
    if ((sign != 1) and (sign != -1)):
        raise ValueError("Invalid sign, should be positive or negative 1")    
    if conv == 'little-endian':
        for i in reversed(range(n)):
            circuit.h(i)
            for j in reversed(range(0,i)):
                circuit.cp(sign*(np.pi/2**(np.abs(i-j))), i, j)    
            #circuit.barrier()        
    elif conv == 'big-endian':
        for i in range(n):
            circuit.h(i)
            for j in range(i+1,n):
                circuit.cp(sign*(np.pi/2**(np.abs(j-i))), i, j)
            #circuit.barrier()
    else:
        raise ValueError("Invalid convention")    
    if swap_after:
        for k in reversed(range(n//2)):
            circuit.swap(k, n-1-k)
        #circuit.barrier()
    else:
        pass    
    return circuit

def iqft(circuit : qiskit.QuantumCircuit ,conv : str, swap_before : bool, swap_after : bool, sign : int) -> qiskit.QuantumCircuit:
    '''
    For a given circuit 'circuit', it will apply
    inverse quantum Fourier transform (IQFT).

    Inputs:
    -------

    circuit : (qiskit.QuantumCircuit)  a quantum circuit

    conv : (str)  'little-endian' or 'big-endian' convention for QFT

    swap_before : (bool)  whether to apply swap gates before QFT

    swap_after : (bool)  whether to apply swap gates after QFT

    sign : (int)  sign of the phase

    Outputs:
    --------

    (qiskit.QuantumCircuit) The previous quantum circuit with the inverse QFT applied to it.                

    '''
    n = circuit.num_qubits
    #circuit.barrier()
    if swap_before:
        for k in range(n//2):
            circuit.swap(k, n-1-k)
        #circuit.barrier()
    else:
        pass
    if ((sign != 1) and (sign != -1)):
        raise ValueError("Invalid sign, should be positive or negative 1")    
    if conv == 'little-endian':
        for i in reversed(range(n)):
            circuit.h(i)
            for j in reversed(range(0,i)):
                circuit.cp(sign*(np.pi/2**(i-j)), i, j)    
            #circuit.barrier()        
    elif conv == 'big-endian':
        for i in reversed(range(n)):
            for j in reversed(range(i+1,n)):
                circuit.cp(sign*(np.pi/2**(j-i)), i, j)
            circuit.h(i)
            #circuit.barrier()
    else:
        raise ValueError("Invalid convention")    
    if swap_after:
        for k in reversed(range(n//2)):
            circuit.swap(k, n-1-k)
        #circuit.barrier()
    else:
        pass    
    return circuit
