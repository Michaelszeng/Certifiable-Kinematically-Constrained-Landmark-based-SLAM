import numpy as np

# Parameters
N = 4  # Num time steps
K = 3   # Num landmarks
d = 3   # dimension of space (3D)

SQRT2 = np.sqrt(2)

# Measurement data: maps landmark to {timestamp: measurement} dicts
y_bar = {0: {0: (1,3,0.5), 1: (-SQRT2/2,SQRT2*1.5,0.5), 2: (-1-(1-(1/SQRT2)),1-(1/SQRT2),0.5), 3: (-(SQRT2-1),-SQRT2,0.5)},
         1: {0: (2,3,0), 1: (0,SQRT2*2,0), 2:(-1-(1-(1/SQRT2)),1+1-(1/SQRT2),0), 3: (-(SQRT2-1)-(SQRT2/2),-SQRT2/2,0)},
         2: {0: (2,2,-0.5), 1: (SQRT2/2, SQRT2*1.5,-0.5), 2: (1-(1/SQRT2),1+1-(1/SQRT2),-0.5), 3: (-(SQRT2-1),0,-0.5)}}

# Covariances
Sigma_p = np.linalg.inv(np.eye(d))  # Covariance matrix for position
Sigma_v = np.linalg.inv(np.eye(d))  # Covariance matrix for velocity
Sigma_omega = np.linalg.inv(np.eye(d**2))  # Covariance matrix for angular velocity

def make_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

# Initial guesses:
t_guess = [[0.01,-0.01,0], [-0.02,0.99,0], [0.71,1.71,0], [1.71,1.71,0]]
R_guess = [make_rot_mat(0.01), make_rot_mat(-np.pi/4), make_rot_mat(-np.pi/2), make_rot_mat(-3*np.pi/4)]
v_guess = [[0,1.0,0]]*(N-1)
Omega_guess = [make_rot_mat(0)]*(N-1)
p_guess = [[1.01,3.01,0.5], [2.01,3.01,0], [2.01,1.99,0]]