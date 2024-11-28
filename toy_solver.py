import cvxpy as cp
import numpy as np
import pandas as pd 

# Outer label: landmark number
# Inner label: timestep
measurements = {
    # Landmark number
    0: {
        # Timestep number
        1: np.array([[6,-3,1]]).T,
        2: np.array([[2,-3,1]]).T,
        3: np.array([[-2,-3,1]]).T,
    },
    1: {
        2: np.array([[0,3,-1]]).T,
        3: np.array([[-4,3,-1]]).T,
    },
    2: {
        0: np.array([[9,7,-4]]).T,
        1: np.array([[5,7,-4]]).T,
        3: np.array([[-3,7,-4]]).T,
    },
    3: {
        0: np.array([[1,2,2]]).T,
        2: np.array([[-7,2,2]]).T,
        3: np.array([[-11,2,2]]).T,
    },
    4: {
        0: np.array([[0,4,1]]).T,
        1: np.array([[-4,4,1]]).T,
        2: np.array([[-8,4,1]]).T,
    },
}

# Number of timesteps and number of measurements
N = 1
for lm_meas in measurements.values():
    for timestep in lm_meas.keys():
        N = max(N, timestep + 1)
K = len(measurements)

# [Omega R p t]
dim_x = 21 * N + 3 * K - 9
dim_v = 3 * N - 3

# Velocity, angular velocity, and measurement covariances
cov_v = 1
cov_omega = 1
cov_meas = 1

# Want to minimize tr(QX) + v^TPv
X = cp.Variable((dim_x, dim_x), symmetric=True)
v = cp.Variable(dim_v)
Q = np.zeros((dim_x, dim_x))
P = np.zeros((dim_v, dim_v))

# Add constant linear velocity objective
P[0:dim_v, 0:dim_v] += cov_v * np.identity(dim_v)
P[3:dim_v, 0:dim_v-3] -= cov_v * np.identity(dim_v-3)
P[0:dim_v-3, 3:dim_v] -= cov_v * np.identity(dim_v-3)
P[3:dim_v-3, 3:dim_v-3] += cov_v * np.identity(dim_v-6)

# Add constant angular velocity objective
Q[0:9*N-9, 0:9*N-9] += cov_omega * np.identity(9*N-9)
Q[9:9*N-9, 0:9*N-18] -= cov_omega * np.identity(9*N-18)
Q[0:9*N-18, 9:9*N-9] -= cov_omega * np.identity(9*N-18)
Q[9:9*N-18, 9:9*N-18] += cov_omega * np.identity(9*N-27)

for k, lm_meas in measurements.items():
    for t, meas in lm_meas.items():
        t_idx = 9*(N+t-1)
        lm_idx = 18*N-9+3*k
        trans_idx = 18*N-9+3*(K+t)

        # Add rotation objective
        for i in range(3):
            Q[t_idx+3*i:t_idx+3*(i+1), t_idx+3*i:t_idx+3*(i+1)] += cov_meas * (meas @ meas.T)

        # Add rotation to landmark objective
        for i in range(3):
            Q[lm_idx+i, t_idx+3*i:t_idx+3*(i+1)] -= cov_meas * meas.flatten()
            Q[t_idx+3*i:t_idx+3*(i+1), lm_idx+i] -= cov_meas * meas.flatten()

        # Add rotation to translation objective
        for i in range(3):
            Q[trans_idx+i, t_idx+3*i:t_idx+3*(i+1)] += cov_meas * meas.flatten()
            Q[t_idx+3*i:t_idx+3*(i+1), trans_idx+i] += cov_meas * meas.flatten()

        # Add landmark to translation objective
        Q[lm_idx:lm_idx+3, trans_idx:trans_idx+3] -= cov_meas * np.identity(3)
        Q[trans_idx:trans_idx+3, lm_idx:lm_idx+3] -= cov_meas * np.identity(3)

        # Add landmark and translation objectives
        Q[lm_idx:lm_idx+3, lm_idx:lm_idx+3] += cov_meas * np.identity(3)
        Q[trans_idx:trans_idx+3, trans_idx:trans_idx+3] += cov_meas * np.identity(3)

# PSD constraint
constraints = [X >> 0]

# Start point has 0 rotation (identity matrix)
for i in range(9*N-9, 9*N):
    for j in range(dim_x):
        A = np.zeros((dim_x, dim_x))
        A[i, j] = A[j, i] = 1 if i == j else 0.5
        if i % 9 in {0, 4, 8} and j % 9 in {0, 4, 8} and j < 18*N-9:
            constraints.append(cp.trace(A @ X) == 1)
        elif i % 9 not in {0, 4, 8}:
            constraints.append(cp.trace(A @ X) == 0)

# Start point has 0 translation
for i in range(18*N-9+3*K, 18*N-6+3*K):
    for j in range(dim_x):
        A = np.zeros((dim_x, dim_x))
        A[i,j] = A[j,i] = 1
        constraints.append(cp.trace(A @ X) == 0)

# R^TR=I constraints
for t in range(N):
    for i in range(3):
        for j in range(3):
            A = np.zeros((dim_x, dim_x))
            for k in range(3):
                A[9*(N+t-1)+j+3*k, 9*(N+t-1)+3*i+k] = 1 if j+3*k == 3*i+k else 0.5
                A[9*(N+t-1)+3*i+k, 9*(N+t-1)+j+3*k] = 1 if j+3*k == 3*i+k else 0.5
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
        offsets = [(0, -9), (1, -6), (2, -3)]  # Offsets for R and t indices
        for offset_t, offset_R in offsets:
            A[9*(N+t)+offset_R+j, 18*N+3*(K+t)+offset_t-9] = 0.5  # R_i * t_i
            A[9*(N+t)+offset_R+j, 18*N+3*(K+t)+offset_t-6] = -0.5 # -R_i * t_{i+1}
            A[18*N+3*(K+t)+offset_t-9, 9*(N+t)+offset_R+j] = 0.5
            A[18*N+3*(K+t)+offset_t-6, 9*(N+t)+offset_R+j] = -0.5
        d = np.zeros((dim_v, 1))
        d[3*t+j,0] = 1 # v_i
        constraints.append(cp.trace(A @ X) + d.T @ v == 0)

# Rotation odometry constraints
for t in range(N - 2):
    for j in range(3):
        for i in range(3):
            A = np.zeros((dim_x, dim_x))
            offsets = [(-9, 0), (-8, 3), (-7, 6)]       # Offset for R_t, Omega_t
            next_offsets = [(9, 9), (10, 10), (11, 11)] # Offset for R_{t+2}, Omega_{t+1}
            for offset_r, offset_o in offsets:
                A[9*(N+t)+offset_r+3*j,9*t+offset_o+i] = 0.5
                A[9*t+offset_o+i, 9*(N+t)+offset_r+3*j] = 0.5
            for offset_r, offset_o in next_offsets:
                A[9*(N+t)+offset_r+3*j, 9*t+offset_o+3*i] = -0.5
                A[9*t+offset_o+3*i, 9*(N+t)+offset_r+3*j] = -0.5
            constraints.append(cp.trace(A @ X) == 0)

# Problem definition
prob = cp.Problem(cp.Minimize(cp.trace(Q @ X) + cp.quad_form(v, P)), constraints)
prob.solve(solver=cp.MOSEK)

# Save X as csv
DF = pd.DataFrame(X.value) 
DF.to_csv("toy_solver.csv")

# Reconstruct x
U, S, Vt = np.linalg.svd(X.value, hermitian=True)
x = U[:, 0] * np.sqrt(S[0])
if x[0] < 0:
    x = -x

# Retrieve Omega R p t
ang_vel = x[:9*N-9].reshape((N-1, 3, 3))
ang_pos = x[9*N-9:18*N-9].reshape((N, 3, 3))
landmarks = x[18*N-9:18*N-9+3*K].reshape((K, 3))
lin_pos = x[18*N-9+3*K:].reshape((N, 3))

# Calculate rank of X
rank = np.linalg.matrix_rank(X.value, rtol=1e-1, hermitian=True)

# Print result
np.set_printoptions(threshold=np.inf, suppress=True,
    formatter={'float_kind':'{:0.4f}'.format})
print("\nThe optimal value is", prob.value)
print()

print("The solution X is saved to toy_solver.csv\n")

print("The singular values of X are")
print(S)
print()

print("Angular velocity:")
print(ang_vel)
print()

print("Rotation matrices:")
print(ang_pos)
print()

print("Landmarks:")
print(landmarks)
print()

print("Linear velocity:")
print(v.value)
print()

print("Linear position:")
print(lin_pos)
print()

print("Rank of X is", rank)
print()

