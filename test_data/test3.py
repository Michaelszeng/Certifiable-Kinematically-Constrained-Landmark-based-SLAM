import numpy as np

d = 3   # dimension of space (3D)

# Measurement data: maps landmark to {timestamp: measurement} dicts
y_bar = {
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

N = 1
for lm_meas in y_bar.values():
    for timestep in lm_meas.keys():
        N = max(N, timestep + 1)
K = len(y_bar)


# Covariances
Sigma_p = np.linalg.inv(4*np.eye(d))  # Covariance matrix for position
Sigma_v = np.linalg.inv(np.eye(d))  # Covariance matrix for velocity
Sigma_omega = np.linalg.inv(np.eye(d**2))  # Covariance matrix for angular velocity

def make_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

# Initial guesses:
t_guess = [[0,0,0], [4,0,0], [8,0,0], [12,0,0]]
R_guess = [make_rot_mat(0), make_rot_mat(0), make_rot_mat(0), make_rot_mat(0)]
v_guess = [[4,0,0]]*(N-1)
Omega_guess = [make_rot_mat(0)]*(N-1)
p_guess = [[10,-3,1], [8,3,-1], [9,7,-4], [1,2,2], [0,4,1]]