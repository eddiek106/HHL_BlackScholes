# -*- coding: utf-8 -*-
"""
File: ancillacircuit.py
Author: Eddie Kelly
Date: 2024

This file is part of the Quantum algorithm for linear systems of equations for the multi-dimensional Black-Scholes equations project which was completed as part 
of the thesis https://mural.maynoothuniversity.ie/id/eprint/19288/.

License: MIT License
"""

import os
import sys
import numpy as np
import qiskit
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.circuit.library.arithmetic.polynomial_pauli_rotations import PolynomialPauliRotations
from typing import Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HHL.ancillarotation import eigenvalueextract, rotation, ave_rotation, rotation_l, rotation_r

def int_to_bin(N_int : int ,Num_bits : int) -> str:
    '''
    Converts an integer to its binary representation with a fixed number of bits.

    Inputs:
    -------

    N_int: (int) Integer to be converted to binary.

    Num_bits: (int) Number of bits to be used in the binary representation.
    
    
    Output:
    -------

    Returns the binary representation of the integer N_int with Num_bits bits.
    '''  
    return format(N_int, '0'+str(Num_bits)+'b')  

def eigenvalueextract_range(M : int, t : float):
    return [eigenvalueextract(i,M,t) for i in range(1,M) if i!= M // 2]


def anc_circ(circuit : qiskit.QuantumCircuit, M : int , t : float, C : float,
              abs_eigval_lower : float, abs_eigval_upper : float) -> qiskit.QuantumCircuit:
    '''

    Applies the controlled Pauli Y rotations to the ancilla qubit based on the rotation angles as calculated
    by the function rotation(). Only Y rotations (whose angles/rotations correspond to) eigenvalues that are within the specified
    range (i.e absolute value is between the lower and upper absolute eigenvalues.) are applied, this
    corresponds to the value of C such that it is less than the smallest eigenvalue of the spectrum in 
    terms of absolute value. If one wants to append every possible rotation for all the of the Fourier 
    basis states the value of C will need to be modified such that it is less than the smallest output
    from the rotation function in terms of absolute value. The abs_eigval_lower and abs_eigval_upper
    would be set to zero and infinity repsectively. The function rotation() maps the Fourier basis states
    to the rotation angles.

    Inputs: 
    -------

    circuit: (qiskit.QuantumCircuit) Quantum circuit to which the ancilla circuit is to be appended.

    M: (int) Number of Fourier basis states used in QPE.

    t: (float) The time that is used in the Unitary operator in QPE ; U = e^{iAt}.

    C: (float) Normalization constant for the ancilla qubit.

    abs_eigval_lower: (float) Lower bound for the absolute value of the eigenvalues.

    abs_eigval_upper: (float) Upper bound for the absolute value of the eigenvalues.

    
    Output:
    -------

    Returns the quantum circuit with the controlled Pauli Y rotations appended to it.  
        
    '''
    Num_qubits = circuit.num_qubits                 #You could also just take this to be log2(M), I didnt just to make it more readable the code
    if (2**(Num_qubits-1)) != M:                    # Exclude the ancilla qubit from the control qubits.
        raise ValueError('The number of qubits excluding ancilla qubit in the circuit must be equal to log2(M) where M is the number of Fourier basis states used in QPE.')
    for i in range(1,M):                             
        if (i == M/2) or (i == 0):
            continue                                 # Excluding the basis states 0 and M/2 from the rotation angles.
        elif abs_eigval_upper>np.abs(eigenvalueextract(i,M,t))>abs_eigval_lower:
            circuit.append(RYGate(rotation(i,M,t,C)).control(Num_qubits-1,str(i),int_to_bin(i,Num_qubits-1)[::-1]),list(reversed(\
                range(Num_qubits))))
        else:
             continue   
    return circuit

def anc_circ_polyfit(circuit : qiskit.QuantumCircuit, M : int, t : float, C: float,
                         order : int) -> qiskit.QuantumCircuit:
    '''
    Applies the controlled Pauli Y rotations to the ancilla qubit based on the rotation angles as calculated
    by a Taylor polynomial approximation to the function rotation(). Only Y rotations (whose angles/rotations correspond to)
    eigenvalues that are within the specified range (i.e absolute value is between the lower and upper 
    absolute eigenvalues.) are applied, this corresponds to the value of C such that it is less than the
    smallest eigenvalue of the spectrum in terms of absolute value. 
    If one wants to append every possible rotation for all the of the Fourier 
    basis states the value of C will need to be modified such that it is less than the smallest output
    from the rotation function in terms of absolute value. The abs_eigval_lower and abs_eigval_upper
    would be set to zero and infinity repsectively. The function rotation() maps the Fourier basis states
    to the rotation angles.


    Inputs:
    -------

    circuit: (qiskit.QuantumCircuit) Quantum circuit to which the ancilla circuit is to be appended.

    M: (int) Number of Fourier basis states used in QPE.

    t: (float) The time that is used in the Unitary operator in QPE ; U = e^{iAt}.

    C: (float) Normalization constant for the ancilla qubit.

    abs_eigval_lower: (float) Lower bound for the absolute value of the eigenvalues.

    abs_eigval_upper: (float) Upper bound for the absolute value of the eigenvalues.

    poly_coeff: (list) List of coefficients of the polynomial that approximates the rotation angles for the Fourier basis states.

    Output:
    -------

    Returns the quantum circuit with the controlled Pauli Y rotations appended to it.  
        
    '''
    Num_qubits = circuit.num_qubits                 #You could also just take this to be log2(M), I didnt just to make it more readable the code
    if (2**(Num_qubits-1)) != M:                    # Exclude the ancilla qubit from the control qubits.
        raise ValueError('The number of qubits excluding ancilla qubit in the circuit must be equal to log2(M)')
    
    else : 
        x_list = [i for i in range(1,M) if i!= M/2]
        rotation_list = [rotation(x,M,t,C) for x in x_list]
        coefficients = np.flip(np.polyfit(x_list,rotation_list,order))
        coefficients = coefficients.tolist()   # Convery numpy array of coefficients to list for the PolynomialPauliRotations class.
        polynomial_pauli_rotations = PolynomialPauliRotations(int(np.log2(M)),coefficients,'Y')  # Just Y rotations for HHL  
        circuit.append(polynomial_pauli_rotations,list(range(Num_qubits)))
        return circuit





if __name__ == "__main__":
    import qiskit
    import matplotlib.pyplot as plt
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import Aer, AerSimulator
    from qiskit.circuit.library.arithmetic.polynomial_pauli_rotations import PolynomialPauliRotations
    from ancillarotation import rotation_l, rotation_r
    import itertools

    C = 0.125
    M = 32
    t = 0.5
    order = 5

    x_list = [i for i in range(1,M) if i!= M/2]
    rotation_list = [rotation(i,M,t,C) for i in x_list]
    coeffs = np.flip(np.polyfit(x_list,rotation_list,order))
    coeffs = coeffs.tolist()
    print(coeffs)

    qc = QuantumCircuit(6)
    anc_circ(qc,M,t,C,0,np.inf)
    qc.barrier()
    (anc_circ_polyfit(qc,M,t,C,order)).inverse()
    qc.measure_all()
    print(qc)

    simulator = AerSimulator()
    circ_transpiled = transpile(qc, simulator)
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(circ_transpiled, shots=10000)
    result = job.result()
    all_basis_states = [''.join(seq) for seq in itertools.product("01", repeat = 6)]
    counts = {state : 0 for state in all_basis_states}
    counts.update(result.get_counts())
    print(counts)
    
