# -*- coding: utf-8 -*-
"""
File: matrixmod.py
Author: Eddie Kelly
Date: 2024

This file is part of the Quantum algorithm for linear systems of equations for the multi-dimensional Black-Scholes equations project which was completed as part 
of the thesis https://mural.maynoothuniversity.ie/id/eprint/19288/.

License: MIT License
"""

import numpy as np
import scipy
from scipy.linalg import expm

def nextpowerof2(n : int) -> int:
    '''
    Returns the next power of 2 for a given number n.

    Inputs: 
    -------

    n : (int) Input number

    Outputs:
    --------

    (int)  Next power of 2 of n.

    '''

    return 2**(np.ceil(np.log2(n)))


def triangular_block(A : np.ndarray, C : int, i : int, j : int) -> np.ndarray:
    '''
    Places a lower triangular matrix that is used as submatrix 
    in the construction of the matrix for the ODE problem (which
    is based on a Taylor expansion of the formal solution. It will place
    the lower triangular matrix at the given indices, with the size of
    the nonezero 'triangular components' equal to C.
    The matrix is of the form: 

     \\vec{X}(T) = e^{AT}\\vec{X}_{0}+\left(e^{AT} - I\\right)A^{-1}\\vec{B}_{0}

    Inputs:
    -------

    A : (numpy.ndarray)  Matrix to be modified
    C : (int)  Size of the block
    i : (int)  Row index
    j : (int)  Column index

    Outputs:
    --------

    (numpy.ndarray) : Matrix A with lower triangular block of size C at indices
    i,j.

    '''

    for k in range(0,C-1):
        A[i+k,j+k] = -1/(k+1)
    for z in range(0,C):
        A[i+C-1,j+z] = -1  

         
def triangular_block_modified(A : np.ndarray, C : int, i : int, j : int) -> np.ndarray:
    '''
    Places a lower triangular matrix that is used as submatrix 
    in the construction of the matrix for the ODE problem (which
    is based on a Taylor expansion of the formal solution. It will place
    the lower triangular matrix at the given indices, with the size of
    the nonezero 'triangular components' equal to C. 
    The matrix is of the form: 

    However, the matrix is further modified that for every fraction in the 
    subdiagonal, the denominator is the next power of 2. 

     \\vec{X}(T) = e^{AT}\\vec{X}_{0}+\left(e^{AT} - I\\right)A^{-1}\\vec{B}_{0}

    Inputs:
    -------

    A : (numpy.ndarray)  Matrix to be modified
    C : (int)  Size of the block
    i : (int)  Row index
    j : (int)  Column index

    Outputs:
    --------

    (numpy.ndarray) : Matrix A with lower triangular block of size C at indices
    i,j.

    '''    

    for k in range(0,C-1):
        A[i+k,j+k] = -1/nextpowerof2(k+1)
    for z in range(0,C):
        A[i+C-1,j+z] = -1  

        
def matrix_ode(N : int, C : int) -> np.ndarray:
    '''
    Produces a block matrix that will solve the following matrix ODE:

    \\frac{d\\vec{x}}{dt} = A\\vec{x}+\\vec{B}_{0}

    by returning a block matrix structure that is used to solve the 
    ODE problem using a Taylor expansion of the formal solution.

    \\vec{X}(T) = e^{AT}\\vec{X}_{0}+\left(e^{AT} - I\\right)A^{-1}\\vec{B}_{0}

    It effectively takes an identity matrix of size (C*N +2) and inserts the 
    submatrix defined by the function triangular_block_modified 
    at consecutive positions along the diagonal.

    Inputs:
    -------

    C : (int) Size of the triangular block repeated under the diagonal
              ignoring the diagonal itself.

    N : (int) Number of times the block is repeated.

    Outputs:
    --------

    (numpy.ndarray) : Block matrix structure for the ODE problem as
                                    described above.
    '''

    A = np.eye(C*N+2)
    for i in range(0,N):
        triangular_block(A,C,(C*i)+1,(C*i))    
    return A   


def matrix_ode_mod(N,C):
    '''
    Produces a block matrix that will solve the following matrix ODE:

    \\frac{d\\vec{x}}{dt} = A\\vec{x}+\\vec{B}_{0}

    by returning a  block matrix structure that is used to solve the 
    ODE problem using a Taylor expansion of the formal solution.

    \\vec{X}(T) = e^{AT}\\vec{X}_{0}+\left(e^{AT} - I\\right)A^{-1}\\vec{B}_{0}

    It effectively takes an identity matrix of size and inserts the 
    submatrix defined by the function triangular_block 
    at consecutive positions along the diagonal.

    Inputs:
    -------

    C : (int)  Size of the triangular block repeated under the diagonal
              ignoring the diagonal itself.

    N : (int)  Number of times the block is repeated.

    Outputs:
    --------

    (numpy.ndarray) : Block matrix structure for the ODE problem as
                                    described above.
    ''' 

    A = np.eye(C*N+2)
    for i in range(0,N):
        triangular_block_modified(A,C,(C*i)+1,(C*i))    
    return A   


def matrix_ode_tens(N : int, C : int,  A : np.ndarray, h_t : float) -> np.ndarray:
    '''
    Produces a block matrix that will solve the following matrix ODE:

    \\frac{d\\vec{x}}{dt} = A\\vec{x}+\\vec{B}(t)

    The vector \\vec{B}(t) is assummed to have possible time dependance here, however
    the block matrix structure produced is independent of this. 

    Inputs:
    -------

    N : (int)  Number of times the block is repeated.
    
    C : (int) Size of the triangular block repeated under the diagonal
              ignoring the diagonal itself.

    A : (numpy.ndarray)  Matrix A in the ODE problem.    

    h_t : (float)  Temporal size. 

    Outputs:
    --------

    (numpy.ndarray) : Block matrix structure for the ODE problem as
                                             described above.  

    '''

    dimension = np.shape(A)[0]
    sub_block = -h_t*np.diag([1/i for i in range(1,C+1)],-1)
    sub_block_p = np.kron(sub_block,A)
    identity_p = np.eye(dimension*(C+1))
    s_block_p = sub_block_p+identity_p
    g_block = np.zeros((C+1,C+1))
    g_block[0,:] = 1 
    g_block_p = np.kron(g_block,np.eye(dimension))
    hg_block_p = np.kron(np.diag(np.ones(N),-1),g_block_p)
    lambda_p = np.kron(np.eye(N+1),s_block_p)-hg_block_p
    lambda_p = np.where(lambda_p== -0,0,lambda_p)
    return lambda_p


def matrix_ode_tens_complete(N : int, C : int, A : np.ndarray, h_t : float) -> np.ndarray:
    '''
    Produces a block matrix that will solve the following matrix ODE:

    \\frac{d\\vec{x}}{dt} = A\\vec{x}+\\vec{B}(t)

    The vector \\vec{B}(t) is assummed to have possible time dependance here, however
    the block matrix structure produced is independent of this. 

    Inputs:
    -------

    N : (int)  Number of times the block is repeated.
    
    C : (int)  Size of the triangular block repeated under the diagonal
              ignoring the diagonal itself.

    A : (numpy.ndarray)  Matrix A in the ODE problem.    

    h_t : (float)  Temporal size 

    Outputs:
    --------

    (numpy.ndarray) : Block matrix structure for the ODE problem as
                                             described above.  

    '''

    A_num_rows = np.shape(A)[0]
    orig_matrix = matrix_ode_tens(N,C,A,h_t) 
    orig_matrix_num_rows = np.shape(orig_matrix)[0]
    aux_matrix = np.zeros((orig_matrix_num_rows+A_num_rows,orig_matrix_num_rows+A_num_rows))
    # include submatrix into larger auxilliary matrix
    aux_matrix[0:orig_matrix_num_rows,0:orig_matrix_num_rows] = orig_matrix
    # insert identity at bottom right hand corner
    aux_matrix[-A_num_rows:,-A_num_rows:] = np.eye(A_num_rows)
    for j in range(2,C+3):
        aux_matrix[-A_num_rows: ,-j*A_num_rows:-(j-1)*A_num_rows ] = -np.eye(A_num_rows)
    return aux_matrix    

def matrix_ode_tens_complete_repeat(N : int, C : int, A : np.ndarray, h_t : float, p : int) -> np.ndarray:
    '''
    Produces a block matrix that will solve the following matrix ODE:

    \\frac{d\\vec{x}}{dt} = A\\vec{x}+\\vec{B}(t)

    The vector \\vec{B}(t) is assummed to have possible time dependence here, however
    the block matrix structure produced is independent of this.
    The solution is repeated produced  p times.

    Inputs:
    -------

    N : (int)  Number of times the block is repeated.

    C : (int)  Size of the triangular block repeated under the diagonal
              ignoring the diagonal itself.

    A : (numpy.ndarray)  Matrix A in the ODE problem.

    h_t : (float)  Temporal step size

    p : (int) Number of times the solution is repeated.

    Outputs:
    --------

    (numpy array) : block matrix structure for the ODE problem as
                    described above with repeat of a solution.  
    '''

    K = matrix_ode_tens_complete(N,C,A,h_t)
    K_num_rows = np.shape(K)[0]
    A_num_rows = np.shape(A)[0]
    auxilliary_matrix = np.zeros((K_num_rows+p*(A_num_rows),K_num_rows+p*(A_num_rows)))
    auxilliary_matrix[0:K_num_rows,0:K_num_rows] = K
    for i in range(0,p):
        auxilliary_matrix[K_num_rows + i*A_num_rows: K_num_rows + (i+1)*A_num_rows,
                             K_num_rows + i*A_num_rows: K_num_rows + (i+1)*A_num_rows] = np.eye(A_num_rows) 
        auxilliary_matrix[K_num_rows + i*A_num_rows: K_num_rows + (i+1)*A_num_rows,
                             K_num_rows + (i-1)*A_num_rows: K_num_rows + (i)*A_num_rows] = -np.eye(A_num_rows)
    return auxilliary_matrix    
                                 

def bound_init_vec(N : int, C : int, V : np.ndarray) -> np.ndarray:
    '''
    Produces the initial vector for the ODE problem as described above.

    Assuming a Taylor expansion of \\vec{B}(t) of the form:
     
    \\vec{B}(t) \sim \\vec{B}_{0}+t\\vec{B}_{1}+\\frac{t^{2}}{2!}\\vec{B}_{2}+\\mathcal{O}(t^{3})
    
    Inputs:
    -------

    N : (int)  nNumber of times the block is repeated.
    
    C : (int)  Size of the triangular block repeated under the diagonal
              ignoring the diagonal itself.
    
    V : (numpy.ndarray)  Matrix whose first row encodes the intial position
                   x_{0} and whose remaining rows encode various 
                   orders of the Taylor expansion of \\vec{B(t)}. Namely,
                   V[0,:] = \\vec{X}_{0} 
                   V[i,:] = \\((t^{i-1})/((i-1)!))*\\vec{B}_{i-1} for i > 0
    
    Outputs:
    --------
    
    (numpy.array) : Initial vector for the ODE problem as
                                        described above.
    '''

    if V.shape[0] != C+1:
        raise ValueError("The number of rows in B must be equal to C")
    
    vec = V[1:,:].ravel() 
    convec1 = np.concatenate((np.zeros(V.shape[1]),vec))
    convec2 = np.concatenate([convec1 for i in range(0,N)])
    convec2[:V.shape[1]] = V[0,:]
    matrix_full_size =(V.shape[1])*(N+1)*(C+1)
    convec3 = np.concatenate((convec2,np.zeros(matrix_full_size-len(convec2))))
    return convec3

def Bmatrix_independent(C : int, h_t : float, A : np.ndarray,
                         V_0 : np.ndarray, V_1 : np.ndarray) -> np.ndarray:
    '''
    Effectively produces the matrix V for the case of time dependent
    matrix ODE however fills last C-1 rows with zeros. The first and 
    second row correspond to the initial position and the zeroth order
    of the Taylor expansion of the vector \vec{B}(t) respectively.

    Inputs:
    -------

    C : (int) Size of the triangular block repeated under the diagonal

    h_t : (float) Temporal step size

    A : (numpy.ndarray) Matrix A in the ODE problem.

    V_0 : (numpy.ndarray) Initial position vector.

    V_1 : (numpy.ndarray) Zeroth order of the Taylor expansion of the vector B(t).

    Outputs:
    --------

    (numpy.array) : matrix V for the ODE problem as

    '''

    V =np.zeros((C+1,np.shape(A)[1]))
    V[0,:] = V_0
    V[1,:] = h_t*V_1
    return V

 

def index_function(N : int, d : int, coord : tuple):
    '''
    Converts from the coordinate system along each each dimension to the overall
    index of the linear system to be solved. 

    Inputs:
    -------

    N : (int)  number of grid points along each dimension.

    d : (int)  number of dimensions in the problem.

    coord : (tuple)  array of coordinates along each dimension.

    The ordering of the coord tuple is as follows:

    coord = (j_{d},j_{d-1},...,j_{1}) where each j_{i} \in [1,N-1]

    where j_{i} is the coordinate along the i-th dimension.

    Outputs:
    --------

    (int) : index corresponding to the linear system to be solved.

    '''

    index = 0
    coord = coord[::-1] # Reverse the order of the coordinates made a huge mistake
    for m in range(1,d+1):
        index += ((N-2)**(m-1))*(coord[m-1]-1) # As N is inclusive of the boundary points which do form part of the enumeration use N-2 instead
    return index + 1
    

def extractor(N : int, C : int, A : np.ndarray, solution : np.ndarray)-> np.ndarray:
    '''
    Extracts the solution to the ODE problem from the full solution vector from
    inverting the matrix produced by the function matrix_ode_tens.

    Inputs:
    -------

    N (int) : number of times the block is repeated.
    
    C (int) : size of the triangular block repeated under the diagonal
              ignoring the diagonal itself.

    A (numpy.array) : matrix A in the ODE problem.

    Outputs:
    --------

    (numpy.array) : full solution vector  produced from solving function 
    matrix_ode_tens(N,C,A,h)^{-1} bound_init_vec(N,C,B).                     
    '''

    m = (A.shape[0])*N*C+(A.shape[0])*N
    return solution[(m):m+A.shape[0]]


def analytic_time_ind(A : np.ndarray,  X_0 : np.ndarray, B_0 : np.ndarray, T : float) -> np.ndarray:
    '''
    For the matrix ODE problem with time independent \\vec{B}_{0},

    \\frac{d\\vec{x}}{dt} = A\\vec{x}+\\vec{B}_{0}

    the function returns the analytic solution at time T:

    \\vec{X}(T) = e^{AT}\\vec{X}_{0}+\left(e^{AT} - I\\right)A^{-1}\\vec{B}_{0}

    Inputs:
    -------

    A (numpy.array) : matrix A in the ODE problem.

    X_0 (numpy.array) : vector \\vec{X}_{0} in the ODE problem.

    B_0 (numpy.array) : vector \\vec{B}_{0} in the ODE problem.

    T (float) : time at which the solution is evaluated.

    Outputs:
    --------

    (numpy array) : solution to the ODE problem at time T.

    '''

    expAT = expm(A*T)
    return np.dot(expAT, X_0) + np.dot(expAT - np.eye(np.shape(A)[0]), np.linalg.inv(A)).dot(B_0)


def analytic_time_dep(A : np.ndarray, V : np.ndarray, T : float, t_trunc :
                       int , k_trunc : int) -> np.ndarray:
    '''
    For the matrix ODE problem with time dependent \\vec{B(t)},

    \\frac{d\\vec{x}}{dt} = A\\vec{x}+B(t)\\vec{B(t)}

    the function returns the analytic solution at time T:

    \\vec{X}(T) = \\sum_{l=0}^{\\infty}T^{l}\\left(\\sum_{k=0}^{\\infty}
    \\frac{(AT)^{k}}{(k+l)!}\\right)\\vec{V}_{l}

    Where the following vector V_{l} has been defined:

    \\begin{gather}
    \\vec{V}_{l} = \\begin{cases} 
        \\\vec{x}_{0} & l = 0 \\ \\
        \\vec{B}_{l-1} & l > 0 \\ \\
    \\end{cases}
    \\end{gather} 

    The vectors \\vec{B}_{i} are the various orders in
    the Taylor expansion of \\vec{B}(t):

    \\vec{B}(t) \\approx \vec{B}_{0}+t\\vec{B}_{1}+
    \\frac{t^{2}}{2!}\\vec{B}_{2}+\\mathcal{O}(t^{3}) 

    Inputs:
    -------

    A (numpy.array) : matrix A in the ODE problem.

    V (numpy.array) : matrix whose first column encodes the intial position
                   x_{0} and whose remaining columns encode various 
                   orders of the Taylor expansion of \\vec{B(t)}. Namely,
                   V[:,0] = \\vec{X}_{0} 
                   V[:,i] = \\vec{B}_{i-1} for i > 0

    T (float) : time at which the solution is evaluated.

    t_trunc (int) : upper summation limit (for dummy variable 'l') in the analytic solution,
                    amounting to the number of Taylor expansion terms in the solution for 
                    \\vec{b(t)} with respect to t.

    k_trunc (int) : upper nested summation limit (for dummy variable 'k') in the analytic solution. 

    Outputs:
    --------

    (numpy array) : solution to the ODE problem at time T.

    '''

    sol = np.zeros(np.shape(V)[0])
    for l in range(0,t_trunc+1):
        mat = np.zeros(np.shape(A))
        for k in range(0,k_trunc+1):
            mat = mat + np.dot(np.linalg.matrix_power(A,k),(T**k/np.math.factorial(k+l)))
        sol = sol + np.dot(T**l, np.dot(mat,V[:,l]))
    return sol        


def hermitify_matrix(A : np.ndarray) -> np.ndarray:
    '''
    For a given matrix, it returns the following
    block matrix that turns the matrix A into a Hermitian matrix (under the assumption
    that the matrix A has only REAL valued entries):

    \\begin{pmatrix}
        0 & A\\ \\
        A^{\\dagger} & 0
    \\end{pmatrix}

    Inputs:
    -------

    A (numpy array) : real valued matrix to be submatrix within in larger block matrix

    Outputs:
    --------

    (numpy array) : block matrix composed of A and its conjugate transpose
                                        in the structure described above.   

    '''
    if(np.iscomplex(A)).any == True:
        raise ValueError("Matrix is not real-valued!")
    
    return np.kron(np.array([[0,0],[1,0]]),A)+np.kron(np.array([[0,1],[0,0]]),np.transpose(A))    

if __name__ == "__main__":
    import numpy as np

    n_grid = 30
    print(index_function(n_grid,2,(1,1)))
