import cvxpy as cp
import numpy as np
import sys
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)
from test1 import *

# Variables
t = [cp.Variable((d,)) for _ in range(N)]         # Positions t_i
R = [cp.Variable((d, d)) for _ in range(N)]      # Rotations R_i
v = [cp.Variable((d,)) for _ in range(N)]         # Velocities v_i
Omega = [cp.Variable((d, d)) for _ in range(N)]  # Angular velocities Ω_i
p = [cp.Variable((d,)) for _ in range(K)]         # Landmark positions p_k

# Covariances
Sigma_p = np.eye(d)  # Covariance matrix for position
Sigma_v = np.eye(d)  # Covariance matrix for velocity
Sigma_omega = np.eye(d**2)  # Covariance matrix for angular velocity

# Objective
objective = 0
for k in (range(K)):
    for j in y_bar[k].keys():
        residual = R[j] @ y_bar[k][j] - (t[j] - p[k])
        objective += cp.quad_form(residual, Sigma_p)

for i in range(N - 1):
    objective += cp.quad_form(v[i + 1] - v[i], Sigma_v)
    objective += cp.quad_form(cp.vec(Omega[i + 1] - Omega[i], order='F'), Sigma_omega)

# Constraints
constraints = []
# Constant twist constraints
for i in range(N - 1):
    constraints.append(t[i + 1] == t[i] + v[i])
    constraints.append(R[i + 1] == R[i] @ Omega[i])

# SO(3) constraints
for i in range(N):
    constraints.append(R[i] @ R[i].T == np.eye(d))  # Enforce orthonormality
    # Note: determinant 1 constraint relaxed/dropped

# Solve the optimization
problem = cp.Problem(cp.Minimize(objective), constraints)
problem.solve()

# Display results
print("Optimal objective value:", problem.value)