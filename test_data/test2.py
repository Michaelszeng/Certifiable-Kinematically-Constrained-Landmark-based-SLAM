import numpy as np

# Parameters
N = 10  # Num time steps
K = 5   # Num landmarks
d = 3   # dimension of space (3D)

# Measurement data: maps landmark to {timestamp: measurement} dicts
y_bar = {0: {0: (-1,3,0), 1: (-1.2,1.6,0)}, 
         1: {2: (1.45,3.0,0), 3: (1.3,2.0,0)},
         2: {3: (-1.15,3.85,0), 4: (-1.4,2.6,0)},
         3: {3: (-0.35,4.3,0), 4: (-0.9,3.0,0), 5: (-1.2,2.0,0)},
         4: {6: (0.1,5.3,0), 7: (-0.4,4.2,0), 8: (3.0,1.0,0)}}

# Covariances
Sigma_p = np.eye(d)  # Covariance matrix for position
Sigma_v = np.eye(d)  # Covariance matrix for velocity
Sigma_omega = np.eye(d**2)  # Covariance matrix for angular velocity

def make_rot_mat(theta):
    return [[np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]]

# Initial guesses:
t_guess = [[1,1,0], [1,2.2,0], [1.2,3.2,0], [1.8,4.2,0], [2.4,5.2,0], [3.1,5.8,0], [3.9,6.5,0], [4.9,6.8,0], [6,6.9,0], [7.3,7,0]]
R_guess = [make_rot_mat(0), make_rot_mat(-0.15708), make_rot_mat(-2*0.15708), make_rot_mat(-3*0.15708), make_rot_mat(-4*0.15708), make_rot_mat(-5*0.15708), make_rot_mat(-6*0.15708), make_rot_mat(-7*0.15708), make_rot_mat(-8*0.15708), make_rot_mat(-9*0.15708)]
v_guess = [[0,1.2,0]]*N
Omega_guess = [make_rot_mat(-0.15708)]*N
p_guess = [[0,4,0], [4,5,0], [3,8,0], [4,8,0], [9,8,0]]