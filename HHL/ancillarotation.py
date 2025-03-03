# -*- coding: utf-8 -*-
"""
File: ancillarotation.py
Author: Eddie Kelly
Date: 2024

This file is part of the Quantum algorithm for linear systems of equations for the multi-dimensional Black-Scholes equations project which was completed as part 
of the thesis https://mural.maynoothuniversity.ie/id/eprint/19288/.

License: MIT License
"""

import numpy as np

def sigmoid(x:float) -> float:
    '''
    Returns the sigmoid function evaluated at x.

    Inputs:
    -------

    x: (float) The value at which the sigmoid function is evaluated at.

    Outputs:
    --------
    (float) The sigmoid function evaluated at x.
    '''
    return 1/(1+np.exp(-x))


def eigenvalueextract(l:int, M:int, t: float) -> float:
    '''
    Maps from the Fourier basis states to eigenvalues, assuming that the spectrum could
    have but not necessarily contain positive and negative eigenvalues.   

    Inputs:
    -------

    l: (int) Fourier basis state in the range 1 to M-1.

    M: (int) Number of Fourier Basis states used in QPE.

    t: (float) The time that is used in the Unitary operator
       in QPE ; U = e^{iAt/M}.
        
    Output:
    -------

     (float) Returns the M-bit binary approximation of the eigenvalue corresponding to the 
    particular phase measured (i.e. l).
    '''
    if ((l>M) or (l<0)):
        raise ValueError("Outside range of Fourier basis states")
    if (l==0 or l == M/2):
        raise ValueError("Eigenvalue is not defined for this Fourier basis state at l = 0 or l = M/2")
    else:
        if l < M/2:
            return (2*np.pi*l)/t
        elif l > M/2:
            return (2*np.pi/t)*(l-M)
        
def eigenvalueextract_blended(l : float, M : int, t : float, transition_width : float = 2) -> float:
    '''
    Maps from the Fourier basis states to eigenvalues, assuming that the spectrum could
    have but not necessarily contain positive and negative eigenvalues. As the function 
    eigenvalue_extract is piecewise linear, this function is a smooth interpolation between
    the two regimes. 

    Inputs:
    -------

    l: (float) Fourier basis states but can be a float in the range 0 to M-1.

    M: (int) Number of Fourier Basis states used in QPE.

    t: (float) The time that is used in the Unitary operator
       in QPE ; U = e^{iAt/M}.

    transition_width: (float) The width of the transition between the two regimes. Default width of 2.
        
    Output:
    -------

    (float) Returns the M-bit binary approximation of the eigenvalue corresponding to the 
    particular phase measured (i.e. l) with an interpolation between the two regimes.
    '''

    if ((l>M) or (l<0)):
        raise ValueError("Outside range of Fourier basis states")
    else:
        weight = sigmoid((l - M/2) / (transition_width))
        return (1 - weight) * ((2*np.pi*l)/t) + weight * ((2*np.pi/t)*(l-M))



def eigenvalueextract_l(l:int, M:int, t:float) -> float:
    '''
    Maps from the Fourier basis states that correspond to positive eigenvalues, assuming that the spectrum could
    have but not necessarily contain positive and negative eigenvalues.  

    Inputs:
    -------

    l: (int) Fourier basis state in the range 0 to M-1.

    M: (int) Number of Fourier Basis states used in QPE.

    t: (float) The time that is used in the Unitary operator
       in QPE ; U = e^{iAt/M}.           
        
    Outputs:
    --------

    (float) Returns the M-bit binary approximation of the eigenvalue corresponding to the 
    particular phase measured (l).
    '''
    return (2*np.pi*l)/t
    
        
def eigenvalueextract_r(l:int, M:int, t:float) -> float:
    '''
    Maps from the Fourier basis states that correspond to negative eigenvalues, assuming that the spectrum could
    have but not necessarily contain positive and negative eigenvalues.   

    Inputs:
    -------

    l: (int) Fourier basis state in the range 0 to M-1.
    M: (int) Number of Fourier Basis states used in QPE.
    t: (float) The time that is used in the Unitary operator
       in QPE ; U = e^{iAt/M}.           
        
    Outputs:
    --------

    (float) Returns the M-bit binary approximation of the eigenvalue corresponding to the 
    particular phase measured (i.e l).
    '''        
    return (2*np.pi/t)*(l - M)   
    
def ave_eigenvalue(l:int, M:int, t:float) -> float:
    '''
    Maps from the Fourier basis states to eigenvalues, assuming that the spectrum could
    have but not necessarily contain positive and negative eigenvalues. This is an approx
    imation again as this approximation is a smooth interpolation between the positive and
    negative eigenvalues, whereas for the actual mapping there is a discontinuity at M/2.   

    Inputs:
    -------

    l: (int) Fourier basis state in the range 0 to M-1.

    M: (int) Number of Fourier Basis states used in QPE.

    t: (float) The time that is used in the Unitary operator
       in QPE ; U = e^{iAt/M}.
    
    Outputs:
    --------

    (float) Returns the interpolated M-bit binary approximation of the eigenvalue corresponding to the 
    particular phase measured.
    '''
    return eigenvalueextract_l(l,M,t)+eigenvalueextract_r(l,M,t)


def rotation(x:int, M:int, t:float, C:float) -> float:
    '''
    Maps from the Fourier basis states to rotation angles for the
    controlled Pauli Y rotations that will be subsequently perform-
    ed on the ancilla qubit, assuming that the spectrum could
    have but not necessarily contain positive and negative eigenvalues.

    Inputs:
    -------

    x: (int) Fourier basis state in the range 0 to M-1.

    M: (int) Number of Fourier Basis states used in QPE.

    t: (float) The time that is used in the Unitary operator
       in QPE ; U = e^{iAt/M}.

    C: Normalization constant for the ancilla qubit.   
        
    Outputs:
    --------

    (float) Returns the M-bit binary approximation of the rotation angle for the
    eigenvalue corresponding to the  particular phase measured (i.e l).
    '''   
    if ((x>M) or (x<0)):
        raise ValueError("Outside range of Fourier basis states")
    if (x==0 or x == M/2):
        raise ValueError("Eigenvalue is not defined for this Fourier basis state at x = 0 or x = M/2")
    else:
        if x < M/2:
            return 2*np.arcsin(C*t/(2*np.pi*x))
        if x > M/2:
            return 2*np.arcsin((C*t)/(2*np.pi*(x-M)))
        
def rotation_blended(x : float, M : int, t : float, C : float, transition_width : float = 2) -> float:
    '''
    Maps from the Fourier basis states to rotation angles for the
    controlled Pauli Y rotations that will be subsequently perform-
    ed on the ancilla qubit, assuming that the spectrum could
    have but not necessarily contain positive and negative eigenvalues. 
    As the eigenvalue extract function is piecewise linear so will the 
    rotation function, in this instance we use a smooth interpolation of
    the eigenvalue_blend function to avoid the discontinuity in the angle.

    Inputs:
    -------

    x: (float) Fourier basis state in the range 0 to M-1 however can be real valued

    M: (int) Number of Fourier Basis states used in QPE.

    t: (float) The time that is used in the Unitary operator
       in QPE ; U = e^{iAt/M}.

    C: Normalization constant for the ancilla qubit.   
           
    Outputs:
    --------

    (float) Returns the M-bit binary approximation of the rotation angle for the
    eigenvalue corresponding to the  particular phase measured (i.e l).
    '''   
    if ((x>M) or (x<0)):
        raise ValueError("Outside range of Fourier basis states")
    else:
        return 2*np.arcsin(C/eigenvalueextract_blended(x,M,t,transition_width))
        
        

def rotation_l(x:int, M:int, t:float, C:float) -> float:
    '''
    Maps from the Fourier basis states (corresponding to positive
    eigenvalues) to rotation angles for the controlled Pauli Y rotations
    that will be subsequently performed on the ancilla qubit, assuming 
    that the spectrum could have but not necessarily contain positive 
    and negative eigenvalues.

    Inputs:
    -------

    x: (int) Fourier basis state in the range 0 to M-1.

    M: (int) Number of Fourier Basis states used in QPE.

    t: (float) The time that is used in the Unitary operator
       in QPE ; U = e^{iAt/M}.

    C: (float) Normalization constant for the ancilla qubit.   
        
    Outputs:
    --------

    (float) Returns the M-bit binary approximation of the rotation angle for the
    eigenvalue corresponding to the  particular phase measured (i.e l).
    ''' 

    return 2*np.arcsin(C*t/(2*np.pi*x))

def rotation_r(x:int, M:int, t:float, C:float) -> float:
    '''
    Maps from the Fourier basis states (corresponding to negative
    eigenvalues) to rotation angles for the controlled Pauli Y rotations
    that will be subsequently performed on the ancilla qubit, assuming 
    that the spectrum could have but not necessarily contain positive 
    and negative eigenvalues.

    Inputs:
    -------

    x: (int) Fourier basis state in the range 0 to M-1.
    M: (int) Number of Fourier Basis states used in QPE.
    t: (float) The time that is used in the Unitary operator
       in QPE ; U = e^{iAt/M}.
    C: Normalization constant for the ancilla qubit.   

    Outputs:
    --------

    (float) Returns the M-bit binary approximation of the rotation angle for the
    eigenvalue corresponding to the  particular phase measured (i.e l).
    ''' 
    return 2*np.arcsin((C*t)/(2*np.pi*(x-M)))

def ave_rotation(x:int, M:int, t:float, C:float) -> float:
    '''
    Maps from the Fourier basis states to rotation angles for the
    controlled Pauli Y rotations that will be subsequently perform-
    ed on the ancilla qubit using the smooth interpolation to avoid 
    the discontinuity in angle corresponding to the change between
    positive and negative eigenvalues, assuming that the spectrum could
    have but not necessarily contain positive and negative eigenvalues.

    Inputs:
    -------

    x: (int) Fourier basis state in the range 0 to M-1.
    M: (int) Number of Fourier Basis states used in QPE.
    t: (float) The time that is used in the Unitary operator
       in QPE ; U = e^{iAt/M}.
    C: Normalization constant for the ancilla qubit.   
    
    Outputs:
    --------

    (float) Returns the interpolated M-bit binary approximation of the rotation
    angle for the eigenvalue corresponding to the particular 
    phase measured (i.e l).
    '''  

    return 1*(rotation_l(x,M,t,C)+rotation_r(x,M,t,C))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sympy import symbols, asin, pi, series

    M = 32
    t = 1
    C = 0.25

    x_list = [i for i in range(1,M,1) if i!= M/2]
    x_list_ave = np.arange(1,M,1)
    x_list_continuum = np.arange(1,(M-1)+0.05,0.05)
    x_list_l = np.arange(1,M/2,1)
    x_list_r = np.arange((M/2) + 1, M,1)

    eigenvalue_list = [eigenvalueextract(i,M,1) for i in x_list]
    eigenvalue_blend_list = [eigenvalueextract_blended(i,M,1,0.1) for i in x_list_continuum]

    rotation_list_l = [rotation_l(x,M,t,C) for x in x_list_l]
    rotation_list_r = [rotation_r(x,M,t,C) for x in x_list_r]
    rotation_list_ave = [ave_rotation(x,M,t,C) for x in x_list_ave]
    rotation_list = [rotation(x,M,t,C) for x in x_list]
    rotation_list_blend = [rotation_blended(x,M,t,C,1) for x in x_list_continuum]

    # -- NUmpy polyfit.1d() function here -- #

    coefficients = np.polyfit(x_list,rotation_list,6)
    coefficients_test = np.flip(np.polyfit(x_list,[3*i**2 + 2*i + 5 for i in x_list],3))

    polynomial = np.poly1d(coefficients)


    print(coefficients_test)
    plt.plot(x_list, rotation_list, label = 'Data')
    plt.plot(x_list, polynomial(x_list), 'r-', label = 'Fit')

    plt.legend()
    plt.show()

