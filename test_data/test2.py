import numpy as np

"""
CONTAINS ROTATION
"""

# Parameters
N = 4  # Num time steps
K = 3   # Num landmarks
d = 3   # dimension of space (3D)

SQRT2 = np.sqrt(2)

# Measurement data: maps landmark to {timestamp: measurement} dicts
y_bar = {0: {0: np.array([1,3,0.5]), 1: np.array([-SQRT2/2,SQRT2*1.5,0.5]), 2: np.array([-1-(1-(1/SQRT2)),1-(1/SQRT2),0.5]), 3: np.array([-(SQRT2-1),-SQRT2,0.5])},
         1: {0: np.array([2,3,0]), 1: np.array([0,SQRT2*2,0]), 2: np.array([-1-(1-(1/SQRT2)),1+1-(1/SQRT2),0]), 3: np.array([-(SQRT2-1)-(SQRT2/2),-SQRT2/2,0])},
         2: {0: np.array([2,2,-0.5]), 1: np.array([SQRT2/2, SQRT2*1.5,-0.5]), 2: np.array([1-(1/SQRT2),1+1-(1/SQRT2),-0.5]), 3: np.array([-(SQRT2-1),0,-0.5])}}

# Covariances
cov_v=1
cov_omega=1
cov_meas=1

def make_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

# Ground Truths:
t_gt = np.array([[0,0,0], [0,1,0], [0.71,1.71,0], [1.71,1.71,0]])
R_gt = np.array([make_rot_mat(0.01), make_rot_mat(-np.pi/4), make_rot_mat(-np.pi/2), make_rot_mat(-3*np.pi/4)])
v_gt = np.array([[0,1.0,0]]*(N-1))
Omega_gt = ([make_rot_mat(0)]*(N-1))
p_gt = np.array([[1,3,0.5], [2,3,0], [2,2,0]])