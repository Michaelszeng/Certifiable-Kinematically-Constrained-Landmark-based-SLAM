import cvxpy as cp
import numpy as np

# Outer label: landmark number
# Inner label: timestep
measurements = {
    0: { # landmark 0
        0: np.array([[3,3,0]]).T, # timestep 0
        3: np.array([[0,3,0]]).T, # timestep 3
    },
}

N = 4
K = len(measurements)
dim_x = 24 * N + 3 * K - 12

# [v Omega R p t]
x = cp.Variable(dim_x)

cov_v = 1
cov_omega = 1
cov_meas = 1

Q = np.zeros((dim_x, dim_x))

# Add constant linear velocity objective
Q[0:3*N-3, 0:3*N-3] += cov_v * np.identity(3*N-3)
Q[3:3*N-3, 0:3*N-6] -= cov_v * np.identity(3*N-6)
Q[0:3*N-6, 3:3*N-3] -= cov_v * np.identity(3*N-6)
Q[3:3*N-6, 3:3*N-6] += cov_v * np.identity(3*N-9)

print("Q - linear velocity block:")
print(Q[:3*N-3, :3*N-3])
print()

# Add constant angular velocity objective
Q[3*N-3:12*N-12, 3*N-3:12*N-12] += cov_omega * np.identity(9*N-9)
Q[3*N+6:12*N-12, 3*N-3:12*N-21] -= cov_omega * np.identity(9*N-18)
Q[3*N-3:12*N-21, 3*N+6:12*N-12] -= cov_omega * np.identity(9*N-18)
Q[3*N+6:12*N-21, 3*N+6:12*N-21] += cov_omega * np.identity(9*N-27)

print("Q - angular velocity block:")
print(Q[3*N-3:12*N-12, 3*N-3:12*N-12])
print()

for k, lm_meas in measurements.items():
    for t, meas in lm_meas.items():
        # Add rotation objective
        Q[12*N-12+9*t:12*N-12+9*t+3, 12*N-12+9*t:12*N-12+9*t+3] += cov_meas * (meas @ meas.T)
        Q[12*N-12+9*t+3:12*N-12+9*t+6, 12*N-12+9*t+3:12*N-12+9*t+6] += cov_meas * (meas @ meas.T)
        Q[12*N-12+9*t+6:12*N-12+9*t+9, 12*N-12+9*t+6:12*N-12+9*t+9] += cov_meas * (meas @ meas.T)

        # Add rotation to landmark objective
        Q[21*N-12+3*k, 12*N-12+9*t:12*N-12+9*t+3] -= cov_meas * meas.flatten()
        Q[21*N-12+3*k+1, 12*N-12+9*t+3:12*N-12+9*t+6] -= cov_meas * meas.flatten()
        Q[21*N-12+3*k+2, 12*N-12+9*t+6:12*N-12+9*t+9] -= cov_meas * meas.flatten()
        Q[12*N-12+9*t:12*N-12+9*t+3, 21*N-12+3*k] -= cov_meas * meas.flatten()
        Q[12*N-12+9*t+3:12*N-12+9*t+6, 21*N-12+3*k+1] -= cov_meas * meas.flatten()
        Q[12*N-12+9*t+6:12*N-12+9*t+9, 21*N-12+3*k+2] -= cov_meas * meas.flatten()

        # Add rotation to translation objective
        Q[21*N-12+3*K+3*t, 12*N-12+9*t:12*N-12+9*t+3] += cov_meas * meas.flatten()
        Q[21*N-12+3*K+3*t+1, 12*N-12+9*t+3:12*N-12+9*t+6] += cov_meas * meas.flatten()
        Q[21*N-12+3*K+3*t+2, 12*N-12+9*t+6:12*N-12+9*t+9] += cov_meas * meas.flatten()
        Q[12*N-12+9*t:12*N-12+9*t+3, 21*N-12+3*K+3*t] += cov_meas * meas.flatten()
        Q[12*N-12+9*t+3:12*N-12+9*t+6, 21*N-12+3*K+3*t+1] += cov_meas * meas.flatten()
        Q[12*N-12+9*t+6:12*N-12+9*t+9, 21*N-12+3*K+3*t+2] += cov_meas * meas.flatten()

        # Add landmark to translation objective
        Q[21*N-12+3*k:21*N-12+3*k+3, 21*N-12+3*K+3*t:21*N-12+3*K+3*t+3] -= cov_meas * np.identity(3)
        Q[21*N-12+3*K+3*t:21*N-12+3*K+3*t+3, 21*N-12+3*k:21*N-12+3*k+3] -= cov_meas * np.identity(3)

        # Add landmark objective
        Q[21*N-12+3*k:21*N-12+3*k+3, 21*N-12+3*k:21*N-12+3*k+3] += cov_meas * np.identity(3)

        # Add translation objective
        Q[21*N-12+3*K+3*t:21*N-12+3*K+3*t+3, 21*N-12+3*K+3*t:21*N-12+3*K+3*t+3] += cov_meas * np.identity(3)

print("Q - rotation block:")
print(Q[12*N-12:21*N-12, 12*N-12:21*N-12])
print()

print("Q - rotation to landmark block:")
print(Q[12*N-12:21*N-12, 21*N-12:21*N-12+3*K])
print()

print("Q - rotation to translation block:")
print(Q[12*N-12:21*N-12, 21*N-12+3*K:24*N-12+3*K])
print()

print("Q - landmark to translation block:")
print(Q[21*N-12:21*N-12+3*K, 21*N-12+3*K:])
print()

print("Q - landmark block:")
print(Q[21*N-12:21*N-12+3*K, 21*N-12:21*N-12+3*K])
print()

print("Q - translation block:")
print(Q[21*N-12+3*K:, 21*N-12+3*K:])
print()
