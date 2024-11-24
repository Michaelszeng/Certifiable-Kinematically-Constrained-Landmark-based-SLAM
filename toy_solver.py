import cvxpy as cp
import numpy as np
import pandas as pd 

# Outer label: landmark number
# Inner label: timestep
measurements = {
    # Landmark number
    0: {
        # Timestep number
        0: np.array([[10,-3,1]]).T,
        1: np.array([[6,-3,1]]).T,
        2: np.array([[2,-3,1]]).T,
        3: np.array([[-2,-3,1]]).T,
        4: np.array([[-6,-3,1]]).T,
    },
    1: {
        0: np.array([[8,3,-1]]).T,
        1: np.array([[4,3,-1]]).T,
        2: np.array([[0,3,-1]]).T,
        3: np.array([[-4,3,-1]]).T,
        4: np.array([[-8,3,-1]]).T,
    },
    2: {
        0: np.array([[9,7,-4]]).T,
        1: np.array([[5,7,-4]]).T,
        2: np.array([[1,7,-4]]).T,
        3: np.array([[-3,7,-4]]).T,
        4: np.array([[-7,7,-4]]).T,
    },
    3: {
        0: np.array([[1,2,2]]).T,
        1: np.array([[-3,2,2]]).T,
        2: np.array([[-7,2,2]]).T,
        3: np.array([[-11,2,2]]).T,
        4: np.array([[-15,2,2]]).T,
    },
    4: {
        0: np.array([[0,4,1]]).T,
        1: np.array([[-4,4,1]]).T,
        2: np.array([[-8,4,1]]).T,
        3: np.array([[-12,4,1]]).T,
        4: np.array([[-16,4,1]]).T,
    },
    5: {
        0: np.array([[14,-6,5]]).T,
        1: np.array([[10,-6,5]]).T,
        2: np.array([[6,-6,5]]).T,
        3: np.array([[2,-6,5]]).T,
        4: np.array([[-2,-6,5]]).T,
    },
}

N = 5
K = len(measurements)
dim_x = 21 * N + 3 * K - 9
dim_v = 3 * N - 3

# [v Omega R p t]
X = cp.Variable((dim_x, dim_x), symmetric=True)
v = cp.Variable(dim_v)

cov_v = 1
cov_omega = 1
cov_meas = 1

Q = np.zeros((dim_x, dim_x))
P = np.zeros((dim_v, dim_v))

# Add constant linear velocity objective
P[0:3*N-3, 0:3*N-3] += cov_v * np.identity(3*N-3)
P[3:3*N-3, 0:3*N-6] -= cov_v * np.identity(3*N-6)
P[0:3*N-6, 3:3*N-3] -= cov_v * np.identity(3*N-6)
P[3:3*N-6, 3:3*N-6] += cov_v * np.identity(3*N-9)

print("P - linear velocity block:")
print(P)
print()

# Add constant angular velocity objective
Q[0:9*N-9, 0:9*N-9] += cov_omega * np.identity(9*N-9)
Q[9:9*N-9, 0:9*N-18] -= cov_omega * np.identity(9*N-18)
Q[0:9*N-18, 9:9*N-9] -= cov_omega * np.identity(9*N-18)
Q[9:9*N-18, 9:9*N-18] += cov_omega * np.identity(9*N-27)

print("Q - angular velocity block:")
print(Q[:9*N-9, :9*N-9])
print()

for k, lm_meas in measurements.items():
    for t, meas in lm_meas.items():
        # Add rotation objective
        Q[9*N-9+9*t:9*N-9+9*t+3, 9*N-9+9*t:9*N-9+9*t+3] += cov_meas * (meas @ meas.T)
        Q[9*N-9+9*t+3:9*N-9+9*t+6, 9*N-9+9*t+3:9*N-9+9*t+6] += cov_meas * (meas @ meas.T)
        Q[9*N-9+9*t+6:9*N-9+9*t+9, 9*N-9+9*t+6:9*N-9+9*t+9] += cov_meas * (meas @ meas.T)

        # Add rotation to landmark objective
        Q[18*N-9+3*k, 9*N-9+9*t:9*N-9+9*t+3] -= cov_meas * meas.flatten()
        Q[18*N-9+3*k+1, 9*N-9+9*t+3:9*N-9+9*t+6] -= cov_meas * meas.flatten()
        Q[18*N-9+3*k+2, 9*N-9+9*t+6:9*N-9+9*t+9] -= cov_meas * meas.flatten()
        Q[9*N-9+9*t:9*N-9+9*t+3, 18*N-9+3*k] -= cov_meas * meas.flatten()
        Q[9*N-9+9*t+3:9*N-9+9*t+6, 18*N-9+3*k+1] -= cov_meas * meas.flatten()
        Q[9*N-9+9*t+6:9*N-9+9*t+9, 18*N-9+3*k+2] -= cov_meas * meas.flatten()

        # Add rotation to translation objective
        Q[18*N-9+3*K+3*t, 9*N-9+9*t:9*N-9+9*t+3] += cov_meas * meas.flatten()
        Q[18*N-9+3*K+3*t+1, 9*N-9+9*t+3:9*N-9+9*t+6] += cov_meas * meas.flatten()
        Q[18*N-9+3*K+3*t+2, 9*N-9+9*t+6:9*N-9+9*t+9] += cov_meas * meas.flatten()
        Q[9*N-9+9*t:9*N-9+9*t+3, 18*N-9+3*K+3*t] += cov_meas * meas.flatten()
        Q[9*N-9+9*t+3:9*N-9+9*t+6, 18*N-9+3*K+3*t+1] += cov_meas * meas.flatten()
        Q[9*N-9+9*t+6:9*N-9+9*t+9, 18*N-9+3*K+3*t+2] += cov_meas * meas.flatten()

        # Add landmark to translation objective
        Q[18*N-9+3*k:18*N-9+3*k+3, 18*N-9+3*K+3*t:18*N-9+3*K+3*t+3] -= cov_meas * np.identity(3)
        Q[18*N-9+3*K+3*t:18*N-9+3*K+3*t+3, 18*N-9+3*k:18*N-9+3*k+3] -= cov_meas * np.identity(3)

        # Add landmark objective
        Q[18*N-9+3*k:18*N-9+3*k+3, 18*N-9+3*k:18*N-9+3*k+3] += cov_meas * np.identity(3)

        # Add translation objective
        Q[18*N-9+3*K+3*t:18*N-9+3*K+3*t+3, 18*N-9+3*K+3*t:18*N-9+3*K+3*t+3] += cov_meas * np.identity(3)

print("Q - rotation block:")
print(Q[9*N-9:18*N-9, 9*N-9:18*N-9])
print()

print("Q - rotation to landmark block:")
print(Q[9*N-9:18*N-9, 18*N-9:18*N-9+3*K])
print()

print("Q - rotation to translation block:")
print(Q[9*N-9:18*N-9, 18*N-9+3*K:])
print()

print("Q - landmark to translation block:")
print(Q[18*N-9:18*N-9+3*K, 18*N-9+3*K:])
print()

print("Q - landmark block:")
print(Q[18*N-9:18*N-9+3*K, 18*N-9:18*N-9+3*K])
print()

print("Q - translation block:")
print(Q[18*N-9+3*K:, 18*N-9+3*K:])
print()

# PSD constraint
constraints = [X >> 0]

# Start point has 0 rotation (identity matrix)
for i in range(9*N-9, 9*N):
    for j in range(dim_x):
        A = np.zeros((dim_x, dim_x))
        if i == j:
            A[i,j] = 1
            A[j,i] = 1
        else:
            A[i,j] = 0.5
            A[j,i] = 0.5
        if (i % 9 == 0 or i % 9 == 4 or i % 9 == 8) and (j % 9 == 0 or j % 9 == 4 or j % 9 == 8) and j < 18*N-9:
            constraints.append(cp.trace(A @ X) == 1)
        if (i % 9 != 0) and (i % 9 != 4) and (i % 9 != 8):
            constraints.append(cp.trace(A @ X) == 0)

# Start point has 0 translation
for i in range(18*N-9+3*K, 18*N-6+3*K):
    for j in range(dim_x):
        A = np.zeros((dim_x, dim_x))
        A[i,j] = 1
        A[j,i] = 1
        constraints.append(cp.trace(A @ X) == 0)

## TODO: REMOVE. ZERO ANGULAR VELOCITY TEST
#for i in range(9*N-9):
#    for j in range(dim_x):
#        A = np.zeros((dim_x, dim_x))
#        if i == j:
#            A[i,j] = 1
#            A[j,i] = 1
#        else:
#            A[i,j] = 0.5
#            A[j,i] = 0.5
#        if (i % 9 == 0 or i % 9 == 4 or i % 9 == 8) and (j % 9 == 0 or j % 9 == 4 or j % 9 == 8) and j < 9*N-9:
#            constraints.append(cp.trace(A @ X) == 1)
#        elif (i % 9 != 0) and (i % 9 != 4) and (i % 9 != 8):
#            constraints.append(cp.trace(A @ X) == 0)

## TODO: REMOVE. ZERO ROTATION TEST
#for i in range(9*N-9, 18*N-9):
#    for j in range(dim_x):
#        A = np.zeros((dim_x, dim_x))
#        if i == j:
#            A[i,j] = 1
#            A[j,i] = 1
#        else:
#            A[i,j] = 0.5
#            A[j,i] = 0.5
#        if (i % 9 == 0 or i % 9 == 4 or i % 9 == 8) and (j % 9 == 0 or j % 9 == 4 or j % 9 == 8) and j < 18*N-9:
#            constraints.append(cp.trace(A @ X) == 1)
#        elif (i % 9 != 0) and (i % 9 != 4) and (i % 9 != 8):
#            constraints.append(cp.trace(A @ X) == 0)

# R^TR=I constraints
for t in range(N):
    for i in range(3):
        for j in range(3):
            A = np.zeros((dim_x, dim_x))
            for k in range(3):
                A[9*N-9+9*t+j+3*k, 9*N-9+9*t+3*i+k] = 1 if j+3*k == 3*i+k else 0.5
                A[9*N-9+9*t+3*i+k, 9*N-9+9*t+j+3*k] = 1 if j+3*k == 3*i+k else 0.5
            constraints.append(cp.trace(A @ X) == (i == j))

# Omega^TOmega=I constraints
for t in range(N):
    for i in range(3):
        for j in range(3):
            A = np.zeros((dim_x, dim_x))
            for k in range(3):
                A[9*t+j+3*k, 9*t+3*i+k] = 1 if j+3*k == 3*i+k else 0.5
                A[9*t+3*i+k, 9*t+j+3*k] = 1 if j+3*k == 3*i+k else 0.5
            constraints.append(cp.trace(A @ X) == (i == j))

# Translation odometry constraints
for t in range(N - 1):
    for j in range(3):
        A = np.zeros((dim_x, dim_x))
        A[9*N-9+9*t+0+j, 18*N-9+3*K+3*t+0] = 0.5  # R_i[0,0]*t_i[0]
        A[9*N-9+9*t+3+j, 18*N-9+3*K+3*t+1] = 0.5  # R_i[1,0]*t_i[1]
        A[9*N-9+9*t+6+j, 18*N-9+3*K+3*t+2] = 0.5  # R_i[2,0]*t_i[2]
        A[9*N-9+9*t+0+j, 18*N-9+3*K+3*t+3] = -0.5 # -R_i[0,0]*t_{i+1}[0]
        A[9*N-9+9*t+3+j, 18*N-9+3*K+3*t+4] = -0.5 # -R_i[1,0]*t_{i+1}[1]
        A[9*N-9+9*t+6+j, 18*N-9+3*K+3*t+5] = -0.5 # -R_i[2,0]*t_{i+1}[2]

        # Transposes
        A[18*N-9+3*K+3*t+0, 9*N-9+9*t+0+j] = 0.5
        A[18*N-9+3*K+3*t+1, 9*N-9+9*t+3+j] = 0.5
        A[18*N-9+3*K+3*t+2, 9*N-9+9*t+6+j] = 0.5
        A[18*N-9+3*K+3*t+3, 9*N-9+9*t+0+j] = -0.5
        A[18*N-9+3*K+3*t+4, 9*N-9+9*t+3+j] = -0.5
        A[18*N-9+3*K+3*t+5, 9*N-9+9*t+6+j] = -0.5

        d = np.zeros((dim_v, 1))
        d[3*t+j,0] = 1 # v_i[0]
        constraints.append(cp.trace(A @ X) + d.T @ v == 0)

# Rotation odometry constraints
for t in range(N - 2):
    for j in range(3):
        for i in range(3):
            A = np.zeros((dim_x, dim_x))
            A[9*N-9+9*t+0+3*j, 9*t+0+i] = 0.5      # R_t[j,0]*Omega_t[0,i]
            A[9*N-9+9*t+1+3*j, 9*t+3+i] = 0.5      # R_t[j,1]*Omega_t[1,i]
            A[9*N-9+9*t+2+3*j, 9*t+6+i] = 0.5      # R_t[j,2]*Omega_t[2,i]
            A[9*N-9+9*t+18+3*j, 9*t+9+3*i] = -0.5  # R_{t+2}[j,0]*Omega_{t+1}[i,0]
            A[9*N-9+9*t+19+3*j, 9*t+10+3*i] = -0.5 # R_{t+2}[j,1]*Omega_{t+1}[i,1]
            A[9*N-9+9*t+20+3*j, 9*t+11+3*i] = -0.5 # R_{t+2}[j,2]*Omega_{t+1}[i,2]

            # Transposes
            A[9*t+0+i, 9*N-9+9*t+0+3*j] = 0.5
            A[9*t+3+i, 9*N-9+9*t+1+3*j] = 0.5
            A[9*t+6+i, 9*N-9+9*t+2+3*j] = 0.5
            A[9*t+9+3*i, 9*N-9+9*t+18+3*j] = -0.5
            A[9*t+10+3*i, 9*N-9+9*t+19+3*j] = -0.5
            A[9*t+11+3*i, 9*N-9+9*t+20+3*j] = -0.5

            constraints.append(cp.trace(A @ X) == 0)

# Problem definition
prob = cp.Problem(cp.Minimize(cp.trace(Q @ X) + cp.quad_form(v, P)), constraints)
prob.solve(solver=cp.MOSEK)

# Print result
np.set_printoptions(threshold=np.inf, suppress=True,
    formatter={'float_kind':'{:0.4f}'.format})
print("\nThe optimal value is", prob.value)
print(prob.status)
print()
print("A solution X is")
print("Angular velocity part:")
print(X.value[:9*N-9, :9*N-9])
print()
print("Rotation part:")
print(X.value[9*N-9:18*N-9, 9*N-9:18*N-9])
print()
print("A solution v is")
print(v.value)
print()

# Reconstruct x
U, S, Vt = np.linalg.svd(X.value)
x = U[:, 0] * np.sqrt(S[0])
if x[0] < 0:
    x = -x
print("Reconstructed x is")
print(x)
print("Rank of X is", np.linalg.matrix_rank(X.value, tol=1e-6, hermitian=True))
print()

DF = pd.DataFrame(X.value) 
DF.to_csv("test.csv")

