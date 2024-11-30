import numpy as np

# Parameters
N = 2  # Num time steps
K = 1   # Num landmarks
d = 3   # dimension of space (3D)

# Measurement data: maps landmark to {timestamp: measurement} dicts
# NOTE: IT'S VERY IMPORTANT THAT NONE OF THESE VALUES ARE PRECISELY ZERO, OR DRAKE WILL AUTO-REMOVE THE CORRESPONDING VARIABLE, MESSING UP THE CONSTRUCTION OF THE Q MATRIX
y_bar = {0: {0: (1.01,2.01,1e-10), 1: (0.99,1.00,1e-10)}}

# Covariances
Sigma_p = np.linalg.inv(np.eye(d))  # Covariance matrix for position
Sigma_v = np.linalg.inv(np.eye(d))  # Covariance matrix for velocity
Sigma_omega = np.linalg.inv(np.eye(d**2))  # Covariance matrix for angular velocity

def make_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

# Initial guesses:
t_guess = [[0.04,-0.01,0], [-0.07,0.94,0]]
R_guess = [make_rot_mat(0.01), make_rot_mat(0.02)]
v_guess = [[0,1.0,0]]*(N-1)
Omega_guess = [make_rot_mat(0)]*(N-1)
p_guess = [[1.01,2.01,0]]