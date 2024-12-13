import numpy as np

"""
All landmarks are in a 1D subspace. (causes failure)
"""

# Parameters
N = 3  # Num time steps
K = 3   # Num landmarks
d = 3   # dimension of space (3D)

# Measurement data: maps landmark to {timestamp: measurement} dicts
y_bar = {0: {0: np.array([1,4,1]), 1: np.array([1,3,1])},
         1: {0: np.array([-1,4,1]), 1: np.array([-1,3,1]), 2: np.array([-1,2,1])},
         2: {1: np.array([0,3,1]), 2: np.array([0,2,1])}}

# Covariances
cov_v=1
cov_omega=1
cov_meas=1

def make_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

# Ground Truths:
t_gt = np.array([[0,0,0], [0,1,0], [0,2,0]])
R_gt = np.array([make_rot_mat(0), make_rot_mat(0), make_rot_mat(0)])
v_gt = np.array([[0,1,0]]*(N-1))
Omega_gt = np.array([make_rot_mat(0)]*(N-1))
p_gt = np.array([[1,4,1], [-1,4,1], [0,4,1]])