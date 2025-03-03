import numpy as np
import matrixmod as mm

np.set_printoptions(precision=5,linewidth=1000,formatter={'float': '{: 0.3f}'.format})
N=100
C=7
h = 0.1
T = N*h
M = np.diag(np.array([1.65,-0.58,1.83,1.27]),k=0)
V_0 = np.array([2,3,4,5])
V_1 = np.array([-2,3,-6,7])

def print_matrix_vector(matrix, vector):
    for row, val in zip(matrix, vector):
        print(' '.join(format(x, "6.2f") for x in row), " | ", format(val, "6.2f"))

print("This is the B matrix for time independent case")
H1 = mm.Bmatrix_independent(C,h,M,V_0,V_1)
print(H1)



print("This is the A matrix driving evolution")
print(M)

D1 = mm.matrix_ode_tens(N,C,M,h)
print("------------------------")
print("This is N = %d  and C = %d and h = %f" % (N,C,h))
print("The time of evolution is :%f" % T)
print("------------------------")
print("This is the matrix has to be inverted to solve ODE with rhs vec:" )
print("------------------------")
D2 =mm.bound_init_vec(N,C,H1)
# print_matrix_vector(D1,D2)
print("------------------------")
print("This is its shape :%s " % (D1.shape,))
print("------------------------")
print("This is the vector on the right hand side of matrix equation:")
print(np.array2string(D2, formatter={'float_kind': lambda x: "%.2f" % x}))
print("------------------------")
sol = np.linalg.solve(D1,D2)
print("Full output solution is ")
indices = list(range(D1.shape[0]))
sol_values = np.array2string(sol,formatter={'float_kind': lambda x: "%.2f" % x})
##for index, value in zip(indices, sol):
##    print("component%d = %f" % (index, value))

print("------------------------")
print("-**------------------**-")
acc_sol = mm.extractor(N,C,M,sol)
print("The solution from matrix subroutine is:")
print(np.array2string(acc_sol,formatter={'float_kind': lambda x: "%.2f" % x}))
print("------------------------")
print("The actual solution as given for the case  of time independent B is:")
print(np.array2string(mm.analytic_time_ind(M,V_0,V_1,T),formatter={'float_kind': lambda x: "%.2f" % x}))
print("-**------------------**-")
print("------------------------")
print("This is N = %d , C = %d , h = %f and time = %f" % (N,C,h,T))
