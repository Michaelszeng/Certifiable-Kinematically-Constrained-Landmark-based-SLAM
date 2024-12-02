import numpy as np

# Parameters
N = 4  # Num time steps
K = 4   # Num landmarks
d = 3   # dimension of space (3D)

# Measurement data: maps landmark to {timestamp: measurement} dicts
y_bar = {0: {0: (1.01,2.01,0.5), 1: (0.99,0.99,0.5)}, 
         1: {1: (-1.00,3.01,0), 2: (-0.99,2.01,0), 3: (-1.01,0.99,0)},
         2: {0: (-1.01,3.01,0), 1: (-0.99,1.99,0), 2: (-0.99,1.02,0)},
         3: {2: (1.99,3.02,0), 3: (2.01,2.01,0)}}

# Covariances
Sigma_p = np.linalg.inv(np.eye(d))  # Covariance matrix for position
Sigma_v = np.linalg.inv(np.eye(d))  # Covariance matrix for velocity
Sigma_omega = np.linalg.inv(np.eye(d**2))  # Covariance matrix for angular velocity

def make_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

# Initial guesses:
t_guess = [[0.04,-0.01,0], [-0.03,1.1,0], [0.08,2.01,0], [-0.07,2.99,0]]
v_guess = [[0,1.0,0]]*(N-1)
p_guess = [[1.1,2.01,0.5], [-0.99,3.97,0], [-1.01,3.01,0], [1.98,5.02,0]]
R_guess = [make_rot_mat(0.01), make_rot_mat(0.02), make_rot_mat(-0.02), make_rot_mat(0.01)]
Omega_guess = [make_rot_mat(0)]*(N-1)