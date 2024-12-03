import numpy as np

"""
All landmarks are in a 1D subspace. (causes failure)
"""

# Parameters
N = 3  # Num time steps
K = 3   # Num landmarks
d = 3   # dimension of space (3D)

# Measurement data: maps landmark to {timestamp: measurement} dicts
y_bar = {0: {0: (1,4,1), 1: (1,3,1)},
         1: {0: (-1,4,1), 1: (-1,3,1), 2:(-1,2,1)},
         2: {1: (0,3,1), 2: (0,2,1)}}

# Covariances
Sigma_p = np.linalg.inv(np.eye(d))  # Covariance matrix for position
Sigma_v = np.linalg.inv(np.eye(d))  # Covariance matrix for velocity
Sigma_omega = np.linalg.inv(np.eye(d**2))  # Covariance matrix for angular velocity

def make_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

# Initial guesses:
t_guess = [[0.01,-0.01,0], [-0.02,0.99,0], [0.01,2.01,0]]
R_guess = [make_rot_mat(0.01), make_rot_mat(0.0), make_rot_mat(0.0)]
v_guess = [[0,1.0,0]]*(N-1)
Omega_guess = [make_rot_mat(0)]*(N-1)
p_guess = [[1.01,4.01,1.02], [-1.01,4.01,1.04], [0.01,3.99,0.97]]