# -*- coding: utf-8 -*-
"""
File: hhl.py
Author: Eddie Kelly
Date: 2024

This file is part of the Quantum algorithm for linear systems of equations for the multi-dimensional Black-Scholes equations project which was completed as part 
of the thesis https://mural.maynoothuniversity.ie/id/eprint/19288/.

License: MIT License
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '..', '..'))) 
                                                                                        

import numpy as np
from scipy.linalg import expm
from typing import Callable
from matplotlib import pyplot as plt
from terminaltables import AsciiTable 
import itertools


import qiskit
from qiskit import *
from qiskit.visualization import circuit_drawer
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.arithmetic.polynomial_pauli_rotations import PolynomialPauliRotations

from HHL.qft import qft , iqft
from HHL.ancillacircuit import anc_circ, anc_circ_polyfit
from HHL.ancillarotation import rotation
from Matrices.matrixmod import hermitify_matrix

def sign_pic(x : float) -> str:
    '''
    Returns one of the strings '+' or '-' depending on the sign of the input x. It used in the
    circuit drawing to denote the sign of the controlled unitary operator exp(iHt) or exp(-iHt).

    Inputs:
    -------
    x : (float) Input number.

    Outputs:
    --------
    sign_pic: (str) String '+' or '-' depending on the sign of the input x.

    '''
    if x > 0:
        return "+"
    else:
        return "-"

def control_unit(hamiltonian : np.ndarray, circuit : qiskit.QuantumCircuit, control_qubits : list,
                  target_qubits : list, t_0 : float, sign : int, conv : str) -> qiskit.QuantumCircuit:
    '''
    For a given quantum circuit, will apply the controlled unitary operator U=exp((sign)*iH*t) where H is the Hamiltonian
    matrix and t_0 is the time parameter. This quantum circuit is the first of quantum phase estimation 
    (QPE) algorithm. OF CRUCIAL IMPORTANCE, is that t_0 is the total time evolution of the controlled unitary
    U=exp(±iHt_0). We will adopt the convention that the total time evolution of the controlled unitary t_0 in the HHL
    algorithm is expressed as the ratio of two other parameters, t and M. The exact expression is:

                          t_0 = t/M

    where t is some auxilliary time parameter and M is the number of Fourier basis states used for QPE. We refer to
    t_0 as the combined time parameter.

    
    Inputs:
    -------

    hamiltonian: (np.ndarray) Hamiltonian matrix that is real valued and Hermitian.

    circuit: (qiskit.QuantumCircuit) Quantum circuit to which the controlled unitary is to be appended.

    control_qubits: (list) List of control qubits. Qubit zero is reserved for the ancilla qubit and hence
                    should not be included in the list.

    target_qubits: (list) List of target qubits.

    t_0: (float) Combined time parameter.

    sign: (int) Sign of the unitary operator.

    conv: (str) Convention for the control qubits. 'little-endian' or 'big-endian'.

    Outputs:
    --------

    control_unit: (qiskit.QuantumCircuit) Quantum circuit with the controlled unitary operator appended.
    
    '''
    if 0 in control_qubits:

        raise ValueError("Control qubit zero is reserved for the ancilla qubit")
    
    if (sign != 1) and (sign != -1):

        raise ValueError("Sign must be either 1 or -1")
    

    for i in control_qubits:

        circuit.h(i)


    if hamiltonian.shape[0] != 2**(len(target_qubits)):

        raise ValueError("Hamiltonian and number of target qubits are not compatible")
    
    if conv == 'little-endian':

        for k in control_qubits:

            U = expm(2**(k-1)*sign*1j*hamiltonian*t_0)
            circuit.append(UnitaryGate(U,label="exp({}iH(t_0/M)*{})".format(sign_pic(sign),2**(k-1))).control(1),[k] + target_qubits)
            
        return circuit    
    
    elif conv == 'big-endian':
        for k in reversed(control_qubits):

            U  = expm(2**(len(control_qubits)-k)*sign*1j*hamiltonian*t_0)
            circuit.append(UnitaryGate(U,label="exp({}iH(t_0/M)*{})".format(sign_pic(sign),2**(len(control_qubits)-k))).control(1),[k] + target_qubits)

        return circuit    
    
    else:

        raise ValueError("Invalid convention")        

def icontrol_unit(hamiltonian : np.ndarray, circuit : qiskit.QuantumCircuit, control_qubits : list,
                  target_qubits : list, t_0 : float, sign : int, conv : str) -> qiskit.QuantumCircuit:
    '''
    For a given quantum circuit, will apply the controlled unitary operator U=exp((sign)*iH*t_0) where H is the Hamiltonian
    matrix and t_0 is the time parameter. OF CRUCIAL IMPORTANCE, is that t_0 is the total time evolution of the controlled unitary
    U=exp(±iHt_0). We will adopt the convention that the total time evolution of the controlled unitary t_0 in the HHL
    algorithm is expressed as the ratio of two other parameters, t and M. The exact expression is:

                          t_0 = t/M

    where t is an auxilliary time parameter and M is the number of Fourier basis states used for QPE. We refer to
    t_0 as the combined time parameter. This quantum circuit is the first step of the inverse quantum phase estimation
    algorithm assuming that the OPPOSITE sign of the controlled unitary operator U=exp((sign)*iH*t_0) has been applied. i.e.

    control_unit(...,...,sign=-1,...)  <=>  icontrol_unit(...,...,sign=+1,...) 

    OR

    control_unit(...,...,sign=+1,...)  <=>  icontrol_unit(...,...,sign=-1,...) 

    (IQPE) algorithm.

    
    Inputs: 
    -------

    hamiltonian: (np.ndarray) Hamiltonian matrix that is real-valued and Hermitian.

    circuit: (qiskit.QuantumCircuit) Quantum circuit to which the controlled unitary is to be appended.

    control_qubits: (list) List of control qubits. Qubit zero is reserved for the ancilla qubit and hence
                    should not be included in the list.

    target_qubits: (list) List of target qubits.

    t_0: (float) Combined time parameter.

    sign: (int) Sign of the unitary operator.

    conv: (str) Convention for the control qubits. 'little-endian' or 'big-endian'.

    Outputs:
    --------
    
    control_unit: (qiskit.QuantumCircuit) Quantum circuit with the controlled unitary operator appended.

    '''
    if 0 in control_qubits:

        raise ValueError("Control qubit zero is reserved for the ancilla qubit")
    
    if (sign != 1) and (sign != -1):

        raise ValueError("Sign must be either 1 or -1")
    

    if hamiltonian.shape[0] != 2**(len(target_qubits)):

        raise ValueError("Hamiltonian and number of target qubits are not compatible")
    
    if conv == 'little-endian':
        for k in reversed(control_qubits):

            U = expm(2**(k-1)*sign*1j*hamiltonian*t_0)
            circuit.append(UnitaryGate(U,label="exp({}iH(t_0/M)*{})".format(sign_pic(sign),2**(k-1))).control(1),[k] + target_qubits)

    elif conv == 'big-endian':
        for k in control_qubits:

            U  = expm(2**(len(control_qubits)-k)*sign*1j*hamiltonian*t_0)
            circuit.append(UnitaryGate(U,label="exp({}iH(t_0/M)*{})".format(sign_pic(sign),2**(len(control_qubits)-k))).control(1),[k] + target_qubits)    

    else:

        raise ValueError("Invalid convention") 
    

    for i in control_qubits:

        circuit.h(i)

    #circuit.barrier()    
    return circuit

def hhl(hamiltonian : np.ndarray, circuit : qiskit.QuantumCircuit, target_qubits : list, qpe_qubits: list,
          M : float, t : float, C : float, abs_eigval_lower : float, abs_eigval_upper : float)  -> qiskit.QuantumCircuit:
    '''
    For a given quantum circuit, will append the HHL algorithm circuit to the given circuit. OF CRUCIAL IMPORTANCE, we
    only consider real-valued Hamiltonian matrices and "real-valued" amplitudes for the quantum states. (By "real valued",
    the relative phase difference between amplitudes is strictly ±1, no arbitray complex relative phases between amplitudes are permitted).  

    Inputs:
    -------

    hamiltonian: (np.ndarray) Hamiltonian matrix that is real-valued and Hermitian.

    circuit: (qiskit.QuantumCircuit) Quantum circuit to which the HHL algorithm circuit is to be appended.

    target_qubits: (list) List of target qubits.

    qpe_qubits: (list) List of qubits for quantum phase estimation.

    M: (float) Number of Fourier basis states used for QPE

    t: (float) Auxilliary time parameter for unitary operator.

    C: (float) Normalization constant for the ancilla qubit.

    abs_eigval_lower: (float) Lower bound on the absolute value of the eigenvalues of the Hamiltonian.

    abs_eigval_upper: (float) Upper bound on the absolute value of the eigenvalues of the Hamiltonian.

    Outputs:
    -------

    circuit: (qiskit.QuantumCircuit) Quantum circuit with the HHL algorithm circuit appended.
    
    ''' 

    circuit = control_unit(hamiltonian, circuit, qpe_qubits, target_qubits,t/M,1,'big-endian') 
    
    circuit.compose(qft(QuantumCircuit(len(qpe_qubits)),'big-endian',False,False,-1),qpe_qubits,inplace=True)
    
    circuit.compose(anc_circ(QuantumCircuit(len(qpe_qubits)+1),M,t,C,abs_eigval_lower,abs_eigval_upper),[0]+qpe_qubits,inplace=True)
    
    circuit.compose(iqft(QuantumCircuit(len(qpe_qubits)),'big-endian',False,False,1),qpe_qubits,inplace=True)
    
    circuit = icontrol_unit(hamiltonian, circuit, qpe_qubits, target_qubits,t/M,-1,'big-endian')  
    
    return circuit

def hhl_poly_pauli_rot(hamiltonian : np.ndarray, circuit : qiskit.QuantumCircuit, target_qubits : list, qpe_qubits: list,
          M : float, t : float, C : float, order : int)  -> qiskit.QuantumCircuit:
    '''
    For a given quantum circuit, will append the HHL algorithm circuit to the given circuit

    Inputs:
    -------

    hamiltonian: (np.ndarray) Hamiltonian matrix.

    circuit: (qiskit.QuantumCircuit) Quantum circuit to which the HHL algorithm circuit is to be appended.

    target_qubits: (list) List of target qubits.

    qpe_qubits: (list) List of qubits for quantum phase estimation.

    M: (float) Number of Fourier basis states used for QPE

    t: (float) Auxilliary time parameter for the unitary operator.

    C: (float) Normalization constant for the ancilla qubit.

    order: The order of the Taylor polynomial approximating the 2arcsin(C/λ) function in the anc_circ_polyfit function.

    Outputs:
    --------

    circuit: (qiskit.QuantumCircuit) Quantum circuit with the HHL algorithm circuit appended.
    
    ''' 

    circuit = control_unit(hamiltonian, circuit, qpe_qubits, target_qubits,t/M,1,'big-endian') 
    
    circuit.compose(qft(QuantumCircuit(len(qpe_qubits)),'big-endian',False,False,-1),qpe_qubits,inplace=True)
    
    circuit.compose(anc_circ_polyfit(QuantumCircuit(len(qpe_qubits)+1),M,t,C,order=6),[0]+qpe_qubits,inplace=True) 
    
    circuit.compose(iqft(QuantumCircuit(len(qpe_qubits)),'big-endian',False,False,1),qpe_qubits,inplace=True)
    
    circuit = icontrol_unit(hamiltonian, circuit, qpe_qubits, target_qubits,t/M,-1,'big-endian')  
    
    return circuit

class HHL_Solver:
    def __init__(self, hamiltonian : np.ndarray, M : float, t : float, C : float, abs_eigval_lower : float,
                  abs_eigval_upper : float, alt_circ : qiskit.QuantumCircuit = None):
        
        if (np.iscomplex(hamiltonian)).any():
            raise ValueError("Hamiltonian must be real-valued!")

        if (np.transpose(hamiltonian) != hamiltonian).all():

            raise ValueError("Hamiltonian must be Hermitian (assuming Hamiltonian is real valued)")
        
        if not np.isclose(np.log2(hamiltonian.shape[0]),round(np.log2(hamiltonian.shape[0]))):

            raise ValueError("Hamiltonian dimension must be a power of 2 as it is is acting on qubits")
        
        if not np.isclose(np.log2(M),round(np.log2(M))):

            raise ValueError("M must be a power of 2 as it is the number of Fourier basis states used for QPE")
        
        if  (C < 0) or (C > 1):

            raise ValueError("C must be between 0 and 1 as it is ancilla normalization")
        
        self.hamiltonian = hamiltonian
        self.M = M
        self.t = t
        self.C = C
        self.t_0 = t/M
        self.abs_eigval_lower = abs_eigval_lower
        self.abs_eigval_upper = abs_eigval_upper
        self.alt_circ = alt_circ

    def circuitpic(self, init_state : np.ndarray, all_rot : bool = True,  decom_level : int = 0,
                    style_ : str = 'mpl', meas_sty : str = 'stdmeas', second_state : np.ndarray = None) -> qiskit.QuantumCircuit:
        '''
        For a given HHL_Solver object, will return the quantum circuit for the HHL algorithm as
        an image.

        Inputs:
        -------
        init_state: (np.ndarray) The quantum state representation of the vector that will be inverted.

        all_rot: (bool) If True, the circuit will rotate by every possible output from the eigenvalue_extract_function.

        decom_level: (int) Level of decomposition for the circuit, 0 for no decomposition.

        style: (str) Style of the image. 'mpl' for Matplotlib, 'latex_source' for LaTeX code to generate the circuit with Tikz, 'text' for ASCII text.

        meas_sty: (str) Nature of how the solution is measured. 'stdmeas' for measurement of solution in standard computational
                        basis. 'overlapmeas' for mesurement of overlap of solution state with another predetermined second quantum
                        state 'second_state'.

        second_state: (np.ndarray) The second quantum state required if the 'overlap' option is selected for 'meas_sty'.

        Outputs:
        -------

        circuit: (qiskit.QuantumCircuit) Image of quantum circuit for the HHL algorithm.

        '''
        if (init_state.shape[0] != (self.hamiltonian).shape[0]): 

            raise ValueError("Initial state and Hamiltonian are not compatible!")
        
        if second_state is not None:
            if (second_state.shape[0] != (init_state).shape[0]):

                raise ValueError("Second state and initial state are not compatible for overlap measurement!")
                        
        if all_rot :

            if np.abs(self.C) > ((2*np.pi)/(self.t_0)):

                raise ValueError("C must be less than or equal to 2*pi/(t_0)\n"
                "in order to avoid undefined values from arcsin function in\n "
                "ancilla circuit from rotating by every possible output from\n"
                "from the eigenvalue_extract_function!")
            
            num_qpe = int(np.round(np.log2(self.M)))
            num_state = int(np.round(np.log2((init_state).shape[0])))
            qpe_qubits = [i for i in range(1,num_qpe+1)]
            target_qubits = [i for i in range(num_qpe+1,num_qpe+num_state+1)]

            if meas_sty == 'stdmeas':

                circuit = QuantumCircuit(num_qpe+num_state+1,num_state+1) 
                circuit.initialize(init_state,target_qubits,normalize=True)
                circuit = hhl(self.hamiltonian,circuit,target_qubits,qpe_qubits,self.M,self.t,self.C,0,np.inf) #By default, no restriction on the absolute values
                # circuit.barrier()                                                                              #for which we rotate by      
                circuit.measure(0,0)
                # circuit.barrier()
                class_reg = list(range(num_state))
                class_reg = [i+1 for i in class_reg]
                circuit.measure(reversed(target_qubits),class_reg)  # Might need to change how we measure here

                for _ in range(decom_level):

                    circuit = circuit.decompose()

                if  style_ == 'mpl':

                    circuit.draw(output='mpl')
                    plt.show()    

                elif style_ == 'text':           

                    print(circuit)

                elif style_ == 'latex_source': 

                    print("~~~~~~~~~~~~~~~~\n\n")
                    print("\nLatex source code for the circuit is:")
                    print("~~~~~~~~~~~~~~~~\n\n")
                    print(circuit.draw(output='latex_source'))

            elif meas_sty == 'overlapmeas':

                if second_state is None:

                    raise ValueError("Second state must be provided for overlap measurement")
                
                                     

                circuit = QuantumCircuit(num_qpe+2*num_state+1+1,2)           # Two qubits measured only, overlap and post-selection of ancilla
                circuit.initialize(init_state,target_qubits,normalize=True)   # Initialize the first state
                circuit.initialize(second_state,[x+ num_state for x in target_qubits],normalize=True)           #Initialize the second state
                circuit = hhl(self.hamiltonian,circuit,target_qubits,qpe_qubits,self.M,self.t,self.C,0,np.inf)  # HHL with no bounds on eigenvalues here
                # circuit.barrier()   
                circuit.measure(0,0)                # Measure rotation ancilla qubit for post-selection and write to classical bit 0
                circuit.h(num_qpe+2*num_state+1)    # Implement SWAP test

                for k in range(1,num_state+1):

                    circuit.cswap(num_qpe+2*num_state+1,num_qpe+k,num_qpe+num_state+k)

                circuit.h(num_qpe+2*num_state+1)
                circuit.measure(num_qpe+2*num_state+1,1)    # Measure overlap qubit and write to classical bit 1
                # circuit.barrier()

                for _ in range(decom_level):

                    circuit = circuit.decompose() 

                if  style_ == 'mpl':

                    circuit.draw(output='mpl')

                elif style_ == 'text':           

                    print(circuit)

                elif style_ == 'latex_source': 

                    print("~~~~~~~~~~~~~~~~\n\n")
                    print("\nLatex source code for the circuit is:")
                    print("~~~~~~~~~~~~~~~~\n\n")
                    print(circuit.draw(output='latex_source'))

        else:

            raise ValueError("The circuitpic function only works with all_rot = True, for now ")

    def solve_by_qasm(self, init_state : np.ndarray, all_rot : bool = True,  meas_sty : str = 'stdmeas',
                       shots_ : int = 10000  ,data_print : str = 'comprehensive', plot_histo : bool = True, second_state : np.ndarray = None) -> qiskit.QuantumCircuit:
        '''
        For a given HHL_Solver object, will return the QASM counts for the given
        measurement style.

        Inputs:
        -------

        init_state : (np.ndarray) 

        all_rot : (bool) If True, the circuit will rotate by every possible output from the eigenvalue_extract_function.

        meas_style : (str) For standard measurement of amplitudes in computational basis
                           let meas_style='stdmeas'. For measuring the overlap of the out
                           -put of HHL algorithm with a second quantum state 'second
                           state', let meas_style='overlapmeas'. This is achieved using 
                           the multi qubit SWAP test.

        shots_ : (int) Number of runs for the circuit to be ran and measured.     

        plot_histo : (bool) If true will produce a histogram of the counts measured.                                

        second_state : (np.ndarray) Second quantum state for which the overlap will be measured. 

        Outputs:
        --------

        circuit: (qiskit.QuantumCircuit) Image of quantum circuit for the HHL algorithm.

        '''
        if (init_state.shape[0] != (self.hamiltonian).shape[0]): 
            raise ValueError("Initial state and Hamiltonian are not compatible")
        
        if second_state is not None:
            if (second_state.shape[0] != (init_state).shape[0]):

                raise ValueError("Second state and initial state are not compatible for overlap measurement")
            
        if all_rot :
            if np.abs(self.C) > ((2*np.pi)/(self.t)):

                raise ValueError("C must be less than or equal to 2*pi/(t)\n"
                "in order to avoid undefined values from arcsin function in\n "
                "ancilla circuit from rotating by every possible output from\n"
                "from the eigenvalue_extract_function")
            
            num_qpe = int(np.round(np.log2(self.M)))                          # Number of qubits for QPE which is also log2(M), \
            num_state = int(np.round(np.log2((init_state).shape[0])))         # is the number of Fourier Basis States used for QPE
            qpe_qubits = [i for i in range(1,num_qpe+1)]                      # Actual QPE register qubit list
            target_qubits = [i for i in range(num_qpe+1,num_qpe+num_state+1)] # Actual vector register qubit list

            if meas_sty == 'stdmeas': 
                                                                              # Standard measurement of amplitudes in computational basis
                circuit = QuantumCircuit(num_qpe+num_state+1,num_state+1)
                circuit.initialize(init_state,target_qubits,normalize=True)
                circuit = hhl(self.hamiltonian,circuit,target_qubits,qpe_qubits,self.M,self.t,self.C,0,np.inf)
                circuit.measure(0,0)
                # circuit.barrier()
                class_reg = list(range(num_state))
                class_reg = [i+1 for i in class_reg]
                circuit.measure(reversed(target_qubits),class_reg)  # Might need to change how we measure here
                # circuit.barrier()

                #----Backend part here----#
                
                simulator = AerSimulator()
                circuit_transpiled = transpile(circuit,simulator)
                backend = Aer.get_backend('qasm_simulator')
                job = backend.run(circuit_transpiled,shots=shots_)
                result = job.result()
                all_basis_states = [''.join(seq) for seq in itertools.product("01", repeat=1+num_state)] # I am adding in all the basis states here
                counts = {state : 0 for state in all_basis_states}
                counts.update(result.get_counts())                                                                             # for all possible measurements. I am just reading off
                                                                                                                               # the ancilla and all target qubits in measurement 
                     
                post_selected_counts = {key: counts[key] for key in sorted(counts) if key[-1] == '1'} # Post-select 1 for ancilla qubit
                ps_counts_vector = list(post_selected_counts.values())                                # Get the post-selected counts
                pseudo_sol_by_qasm = np.sqrt(np.array(ps_counts_vector)/np.sum(ps_counts_vector))     # Corresponding probabililies as pseudo_solution
                if pseudo_sol_by_qasm.size != 0:

                    possible_solution_vec = [f'±{num}' for num in np.round(pseudo_sol_by_qasm,3)]


                # In order to get a fair comparison, I will invert the normalized version 
                # of the vector init_state. I might add later support for non-normalized
                # vector comparisons. I think it may actually be redundant if we examine 
                # the matrix norm of hamiltonian. 

                actual_solution = np.linalg.solve(self.hamiltonian,init_state)
                actual_solution_for_normalised_b = np.linalg.solve(self.hamiltonian,init_state/np.linalg.norm(init_state))
                actual_solution_for_normalised_b_normalised = actual_solution_for_normalised_b/(np.linalg.norm(actual_solution_for_normalised_b))
                
                pseudo_sol_scaled_by_analy = pseudo_sol_by_qasm * np.linalg.norm(actual_solution)
                pseudo_sol_scaled_by_est = pseudo_sol_by_qasm * 0.5 # This is some constant i have not yet determined #

                #------ circuit details ------#
                if data_print == 'comprehensive':

                    table_data = [ 
                        ['Parameters','Values'],
                        ['Time parameter in hamiltonian simulation exp(iAt/M) t:', np.round(self.t,3)],
                        ['Number of Fourier basis states used for QPE         M:', self.M],
                        ['Ancilla normalization constant                      C:', np.round(self.C,3)],
                        ['Range of |λ| that could be rotated in AR             :', f'[{self.abs_eigval_lower},{self.abs_eigval_upper}]']
                    ]
                    table = AsciiTable(table_data)

                    print("      -@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@- ")
                    print("      |   ---------------------------------   | ")
                    print("      |   * * * Classical Information * * *   | ")
                    print("      |   ---------------------------------   | ")
                    print("      -@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-\n")
                    print("The matrix to be inverted                                :\n\n %s \n" %(self.hamiltonian))
                    print("The (un-normalized) input state |b> &  norm |||b>||      : %s, ||.|| = %f\n" %(init_state,np.linalg.norm(init_state)))
                    print("The input state normalized  |~b> =|b>/|||b>||            : %s\n" %(init_state/np.linalg.norm(init_state)))
                    print("     -----------------------------------------------------------------")
                    print("     Defining classical actual solution state as  |x_acc> = A^-1 |b>  ")
                    print("     Defining (un-normalized)  solution state as  |x'> = A^-1 |~b>    ")
                    print("     Defining normalized solution state as        |~x> = |x'>/|||x'>||")
                    print("     -----------------------------------------------------------------\n")
                    print("The actual solution state  |x_acc> & norm |||x_acc>||    : %s, ||.|| = %f\n" %(actual_solution,np.linalg.norm(actual_solution)))
                    print("The (un-normalized) solution state  |x'> & norm |||x'>|| : %s, ||.|| = %f\n" %(actual_solution_for_normalised_b,np.linalg.norm(actual_solution_for_normalised_b)))
                    print("The solution state normalized |~x>                       : %s\n" %(actual_solution_for_normalised_b_normalised))
                    print("           |------------------------|")
                    print("           |Spectral characteristics|")
                    print("           |      of the matrix     |")
                    print("           |------------------------|\n")
                    print("Absolute eigenvalue interval                     |λ| ∈   : [%f,%f]\n"  %(np.min(np.abs(np.linalg.eigvals(self.hamiltonian))),np.max(np.abs(np.linalg.eigvals(self.hamiltonian)))))
                    print("Condition number of the matrix                    κ      : %f\n" %(np.linalg.cond(self.hamiltonian)))
                    print("The Frobenius norm of the matrix              ||.||_F    : %f\n" %(np.linalg.norm(self.hamiltonian)))
                    print("The 2-norm of the matrix                      ||.||_Spec : %f\n\n\n\n\n" %(np.linalg.norm(self.hamiltonian, ord =2)))
                    print("      -@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@- ")
                    print("      |   -------------------------------   | ")
                    print("      |   * * * Quantum Information * * *   | ")
                    print("      |   -------------------------------   | ")
                    print("      -@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@- \n\n")
                    print("Total number of runs of circuit                     : %d\n" %(shots_))
                    print("Total number of successful p.s of ancilla           : %d\n" %(np.sum(ps_counts_vector))) # sum of post-selected counts is just the number of successful post-selections
                    print("Success rate for post selection                     : %f percent \n" %((np.sum(ps_counts_vector)/shots_)*100))

                    if np.sum(ps_counts_vector) == 0:

                        print("! No probabilities to calculate as no successful post-selections !\n")
                        print("! No pseudo solution to calculate from probabilities ! \n")
                        print("Try to increase number of shots or examine ancilla normalization constant C\n")

                    else:
                        print("Counts dictionary                                   : %s\n" %(counts))
                        print("Post-selected counts dictionary                     : %s\n" %(post_selected_counts))
                        print("Counts                                              : %s\n" %(ps_counts_vector))
                        print("Probabilities from the counts                       : %s\n" %(np.round(ps_counts_vector/np.sum(ps_counts_vector),3)))
                        print("Pseudo real-valued solution for normalized input    : %s\n" %(possible_solution_vec))
                        print("----Scaling pseudo real-valued solution by ||x_acc|| which is analytically calculated---------------\n")
                        print("Pseudo real-valued solution                         : %s\n" %(pseudo_sol_scaled_by_analy))
                        print("----Scaling pseudo real-valued solution by estimates of ||x_acc|| from spectral characteristics-----\n")
                        print("Pseudo real-valued solution                         : %s\n" %(pseudo_sol_scaled_by_est))
                        print("Expected counts given actual solution               : %s\n" %(np.round(np.sum(ps_counts_vector)*(actual_solution_for_normalised_b_normalised**2),0)))

                    print("-----------------------")    
                    print("|   Circuit details   |")    
                    print("-----------------------\n")    
                    print(table.table)
            
                elif data_print == 'quantum_estimate_of_sol_exactnorm' : # Just return the pseudo solution vector with actual norm of classical solved solution

                    return pseudo_sol_scaled_by_analy
                
                elif data_print == 'quantum_estimate_of_sol_estnorm' :
                        
                        return pseudo_sol_scaled_by_est
                

            elif meas_sty == 'overlapmeas':

                circuit = QuantumCircuit(num_qpe+2*num_state+1+1,2)
                circuit.initialize(init_state,target_qubits,normalize=True)
                circuit.initialize(second_state,[x+ num_state for x in target_qubits],normalize=True)
                circuit = hhl(self.hamiltonian,circuit,target_qubits,qpe_qubits,self.M,self.t,self.C,0,np.inf)
                circuit.barrier()   
                circuit.measure(0,0)
                circuit.barrier()
                circuit.h(num_qpe+2*num_state+1) # The SWAP test is done at this stage of the circuit
                for k in range(1,num_state+1):
                    circuit.cswap(num_qpe+2*num_state+1,num_qpe+k,num_qpe+num_state+k)
                circuit.h(num_qpe+2*num_state+1)
                circuit.measure(num_qpe+2*num_state+1,1)    
                circuit.barrier()

                #----Backend part here----#
                
                simulator = AerSimulator()
                circuit_transpiled = transpile(circuit,simulator)
                backend = Aer.get_backend('qasm_simulator')
                job = backend.run(circuit_transpiled,shots=shots_)
                result = job.result()
                all_basis_states = [''.join(seq) for seq in itertools.product("01", repeat=1+num_state)] # I am adding in all the basis states here
                counts = {state : 0 for state in all_basis_states}
                counts.update(result.get_counts())

                post_selected_counts = {key: counts[key] for key in sorted(counts) if key[-1] == '1'} # Post-select 1 for ancilla qubit
                ps_counts_vector = list(post_selected_counts.values())
                print(ps_counts_vector)     
                pseudo_sol_by_qasm = np.sqrt(np.array(ps_counts_vector)/np.sum(ps_counts_vector))     # Corresponding probabililies as pseudo_solution
                if pseudo_sol_by_qasm.size != 0:

                    possible_solution_vec = [f'±{num}' for num in np.round(pseudo_sol_by_qasm,3)] # Assuming real valued hamiltonian and initial_state, the solution will be real valued
                                                                                                  # and will be determined modulo a global sign difference ±. If it were a complex solution, it would 
                                                                                                  # be unique up to a global complex phase exp(iθ)                                   
                                           
                # Get the post-selected counts
                # In order to get a fair comparison, I will invert the normalized version 
                # of the vector init_state. I might add later support for non-normalized
                # vector comparisons. I think it may actually be redundant if we examine 
                # the matrix norm of hamiltonian. 

                actual_solution = np.linalg.solve(self.hamiltonian,init_state)
                actual_solution_for_normalised_b = np.linalg.solve(self.hamiltonian,init_state/np.linalg.norm(init_state))
                actual_solution_for_normalised_b_normalised = actual_solution_for_normalised_b/(np.linalg.norm(actual_solution_for_normalised_b))
                
                pseudo_sol_scaled_by_analy = pseudo_sol_by_qasm * np.linalg.norm(actual_solution)
                pseudo_sol_scaled_by_est = pseudo_sol_by_qasm * 0.5 # This is some constant i have not yet determined #

                #------ circuit details ------#
                if data_print == 'comprehensive':

                    table_data = [ 
                        ['Parameters','Values'],
                        ['Time parameter in hamiltonian simulation exp(iAt/M) t:', np.round(self.t,3)],
                        ['Number of Fourier basis states used for QPE         M:', self.M],
                        ['Ancilla normalization constant                      C:', np.round(self.C,3)],
                        ['Range of |λ| that could be rotated in AR             :', f'[{self.abs_eigval_lower},{self.abs_eigval_upper}]']
                    ]
                    table = AsciiTable(table_data)

                    print("      -@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@- ")
                    print("      |   ---------------------------------   | ")
                    print("      |   * * * Classical Information * * *   | ")
                    print("      |   ---------------------------------   | ")
                    print("      -@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-\n")
                    print("The matrix to be inverted                                :\n\n %s \n" %(self.hamiltonian))
                    print("The (un-normalized) input state |b> &  norm |||b>||      : %s, ||.|| = %f\n" %(init_state,np.linalg.norm(init_state)))
                    print("The input state normalized  |~b> =|b>/|||b>||            : %s\n" %(init_state/np.linalg.norm(init_state)))
                    print("The (un-normalized) second state |d> & norm |||d>||     ")
                    print("     -----------------------------------------------------------------")
                    print("     Defining classical actual solution state as  |x_acc> = A^-1 |b>  ")
                    print("     Defining (un-normalized)  solution state as  |x'> = A^-1 |~b>    ")
                    print("     Defining normalized solution state as        |~x> = |x'>/|||x'>||")
                    print("     -----------------------------------------------------------------\n")
                    print("The actual solution state  |x_acc> & norm |||x_acc>||    : %s, ||.|| = %f\n" %(actual_solution,np.linalg.norm(actual_solution)))
                    print("The (un-normalized) solution state  |x'> & norm |||x'>|| : %s, ||.|| = %f\n" %(actual_solution_for_normalised_b,np.linalg.norm(actual_solution_for_normalised_b)))
                    print("The solution state normalized |~x>                       : %s\n" %(actual_solution_for_normalised_b_normalised))
                    print("           |------------------------|")
                    print("           |Spectral characteristics|")
                    print("           |      of the matrix     |")
                    print("           |------------------------|\n")
                    print("Absolute eigenvalue interval                     |λ| ∈   : [%f,%f]\n"  %(np.min(np.abs(np.linalg.eigvals(self.hamiltonian))),np.max(np.abs(np.linalg.eigvals(self.hamiltonian)))))
                    print("Condition number of the matrix                    κ      : %f\n" %(np.linalg.cond(self.hamiltonian)))
                    print("The Frobenius norm of the matrix              ||.||_F    : %f\n" %(np.linalg.norm(self.hamiltonian)))
                    print("The 2-norm of the matrix                      ||.||_Spec : %f\n\n\n\n\n" %(np.linalg.norm(self.hamiltonian, ord =2)))
                    print("      -@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@- ")
                    print("      |   -------------------------------   | ")
                    print("      |   * * * Quantum Information * * *   | ")
                    print("      |   -------------------------------   | ")
                    print("      -@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@- \n\n")
                    print("Total number of runs of circuit                     : %d\n" %(shots_))
                    print("Total number of successful p.s of ancilla           : %d\n" %(np.sum(ps_counts_vector))) # sum of post-selected counts is just the number of successful post-selections
                    print("Success rate for post selection                     : %f percent \n" %((np.sum(ps_counts_vector)/shots_)*100))

                    if np.sum(ps_counts_vector) == 0:

                        print("! No probabilities to calculate as no successful post-selections !\n")
                        print("! No pseudo solution to calculate from probabilities ! \n")
                        print("Try to increase number of shots or examine ancilla normalization constant C\n")

                    else:

                        print("Counts                                              : %s\n" %(ps_counts_vector))
                        print("Probabilities from the counts                       : %s\n" %(np.round(ps_counts_vector/np.sum(ps_counts_vector),3)))
                        print("Pseudo real-valued solution for normalized input    : %s\n" %(possible_solution_vec))
                        print("----Scaling pseudo real-valued solution by ||x_acc|| which is analytically calculated---------------\n")
                        print("Pseudo real-valued solution                         : %s\n" %(pseudo_sol_scaled_by_analy))
                        print("----Scaling pseudo real-valued solution by estimates of ||x_acc|| from spectral characteristics-----\n")
                        print("Pseudo real-valued solution                         : %s\n" %(pseudo_sol_scaled_by_est))
                        print("Expected counts given actual solution               : %s\n" %(np.round(np.sum(ps_counts_vector)*(actual_solution_for_normalised_b_normalised**2),0)))

                    print("-----------------------")    
                    print("|   Circuit details   |")    
                    print("-----------------------\n")    
                    print(table.table)
            
                elif data_print == 'quantum_estimate_of_sol_exactnorm' : # Just return the pseudo solution vector with actual norm of classical solved solution

                    return pseudo_sol_scaled_by_analy
                
                elif data_print == 'quantum_estimate_of_sol_estnorm' :
                        
                        return pseudo_sol_scaled_by_est
 
        else:

            raise ValueError("The solve_by_qasm function only works with all_rot = True, for now ")





#@@@@@@@@@@@@@@@@@@@@@@@@@@@ TEST CODE BELOW @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

                                #  ||  #                        
                                #  ||  #                        
                                #  ||  #                        
                                # ---- #
                                # \  / #
                                #  \/  #

if __name__ == "__main__":
    #---Just some formatting for the terminal---#

    np.set_printoptions(precision=3)

    #-----------------Parameters for HHL circuit size - vector register-----------------#

    # (0             M) (\vec{0}) = (\vec{b})
    # (M^{dagger}    0) (\vec{x}) = (\vec{0})
    #
    #
    
    

    #Components of vector to be solved for
    a = 4
    b = 8 
    vector = np.array([a,b])

    #Matrix definiition

    matrix = np.array([[1,3],[3,1]])  
    #------------Parameters for HHL circuit selection - QPE-----------------#

    t = 2*np.pi*np.linalg.cond(matrix)

    M_prime = int(2**(np.ceil(np.log2(2*np.linalg.cond(matrix)))))

    # If you want to use a specific M value, overwrite between these two comments
    M = 8
    #~~~~~~~           

    M = max(M_prime,M) # Just taking the maximum of the two values.

    qpe_reg_num_qubits = int(np.log2(M))

    vector_reg_num_qubits = int(np.log2(len(matrix)))

    print("---")
    print("The time parameter (as the lower bound)            :",np.round(t,2))
    print("Acceptable range for time parameter                :(%f, %f)"%(np.round(t,1),np.round(np.pi*M,1)))

    # If you want to use a specific t value, overwrite below
    # t = 18
    #~~~~~~~                                        

    print("The time parameter used                            :",np.round(t))
    print("The lower bound for number of Fourier basis states :",M_prime)
    print("Number of Fourier basis states used                :",M)
    print("Number of qubits for QPE                           :",qpe_reg_num_qubits)

    if (t > np.pi*M):
        raise ValueError("Time parameter is too large for the number of qubits.")

    #-----------------Parameters for HHL circuit  - qubit lists-----------------#
    #-----------------Parameters for HHL circuit  - qubit lists-----------------#
    print("---")

    ancilla_qubit = [0]
    qpe_qubits = [1,2,3]
    first_vector_qubits = [4]

    print("Total number of qubits          :",1+qpe_reg_num_qubits + 1 + 1)
    print("These are ancilla qubit         :",ancilla_qubit)
    print("These are qpe qubits            :",qpe_qubits)
    print("These are first vector qubits   :",first_vector_qubits)

    #-----------------Parameters for HHL circuit - ancilla normalisation and eigenvalue tolerances-----------------#

    eps = 0
    C = ((2*np.pi)/t) - eps 

    abs_eigval_lower = 0       # Not actually imposing any lower bound.
    abs_eigval_upper = np.inf  # Not actually imposing any upper bound.

    print("---")
    print("Normalisation constant 'C' of ancilla :",np.round(C,3))
    print(M_prime)

    circ = QuantumCircuit(1+3+1,2)
    circ.initialize(vector,[4],normalize=True)
    hhl_poly_pauli_rot(matrix,circ,[4],[1,2,3],M,t,C,3)
    circ.barrier()
    circ.measure(0,0)
    circ.measure(4,1)
    print(circ)

    simulator = AerSimulator()
    circ_transpiled = transpile(circ,simulator)
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(circ_transpiled,shots=10000)
    result = job.result()
    all_basis_states = [''.join(seq) for seq in itertools.product("01", repeat=2)] # I am adding in all the basis states here
    counts = {state : 0 for state in all_basis_states}
    counts.update(result.get_counts())                                                                             # for all possible measurements. I am just reading off
    print(counts)
    post_selected_counts = {key: counts[key] for key in sorted(counts) if key[-1] == '1'}
    print(post_selected_counts)
 