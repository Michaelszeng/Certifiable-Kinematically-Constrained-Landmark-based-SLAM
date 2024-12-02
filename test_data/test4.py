import numpy as np

d = 3   # dimension of space (3D)

# Measurement data: maps landmark to {timestamp: measurement} dicts
y_bar = {
    0: {
        0: np.array([[-19.7918,-9.0817,-13.5555]]).T,
        1: np.array([[-21.7918,-15.1428,-8.0646]]).T,
        2: np.array([[-23.7918,-17.6463,-0.2788]]).T,
        3: np.array([[-25.7918,-15.9216,7.7157]]).T,
        4: np.array([[-27.7918,-10.4306,13.7768]]).T,
    },
    1: {
        0: np.array([[-10.6826,-1.0139,0.0598]]).T,
        1: np.array([[-12.6826,-1.3482,-0.3073]]).T,
        2: np.array([[-14.6826,-1.8212,-0.4581]]).T,
        3: np.array([[-16.6826,-2.3063,-0.3521]]).T,
        4: np.array([[-18.6826,-2.6733,-0.0178]]).T,
    },
    2: {
        0: np.array([[13.0070,-11.2591,3.7143]]).T,
        1: np.array([[11.0070,-8.3935,7.9802]]).T,
        2: np.array([[9.0070,-3.7789,10.2418]]).T,
        3: np.array([[7.0070,1.3483,9.8931]]).T,
        4: np.array([[5.0070,5.6142,7.0275]]).T,
    },
    3: {
        0: np.array([[-2.5739,-6.8515,-7.5816]]).T,
        1: np.array([[-4.5739,-10.2244,-4.0061]]).T,
        2: np.array([[-6.5739,-11.3577,0.7768]]).T,
        3: np.array([[-8.5739,-9.9476,5.4855]]).T,
        4: np.array([[-10.5739,-6.3722,8.8584]]).T,
    },
    4: {
        0: np.array([[-19.9287,7.0518,7.3306]]).T,
        1: np.array([[-21.9287,9.2723,1.9565]]).T,
        2: np.array([[-23.9287,8.5083,-3.8078]]).T,
        3: np.array([[-25.9287,4.9645,-8.4179]]).T,
        4: np.array([[-27.9287,-0.4095,-10.6384]]).T,
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
t_guess = [
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 1.0],
    [4.0, -0.49999999999999994, 1.8660254037844388],
    [6.0, -1.3660254037844386, 2.366025403784439],
    [8.0, -2.3660254037844384, 2.366025403784439],
]
R_guess = [
    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    np.array([[1.0, 0.0, 0.0], [0.0, 0.8660254037844387, -0.49999999999999994], [0.0, 0.49999999999999994, 0.8660254037844387]]),
    np.array([[1.0, 0.0, 0.0], [0.0, 0.5000000000000001, -0.8660254037844386], [0.0, 0.8660254037844386, 0.5000000000000001]]),
    np.array([[1.0, 0.0, 0.0], [0.0, 2.1460752085336256e-16, -1.0], [0.0, 1.0, 2.0717043678169387e-16]]),
    np.array([[1.0, 0.0, 0.0], [0.0, -0.4999999999999998, -0.8660254037844388], [0.0, 0.8660254037844388, -0.4999999999999998]]),
]
v_guess = [
    [2, 0, 1],
    [2, 0, 1],
    [2, 0, 1],
    [2, 0, 1],
]
Omega_guess = [
    np.array([[1.0, 0.0, 0.0], [0.0, 0.8660254037844387, -0.49999999999999994], [0.0, 0.49999999999999994, 0.8660254037844387]]),
    np.array([[1.0, 0.0, 0.0], [0.0, 0.8660254037844387, -0.49999999999999994], [0.0, 0.49999999999999994, 0.8660254037844387]]),
    np.array([[1.0, 0.0, 0.0], [0.0, 0.8660254037844387, -0.49999999999999994], [0.0, 0.49999999999999994, 0.8660254037844387]]),
    np.array([[1.0, 0.0, 0.0], [0.0, 0.8660254037844387, -0.49999999999999994], [0.0, 0.49999999999999994, 0.8660254037844387]]),
]
p_guess = [
    [-19.79184322307102, -9.08172745429074, -13.555538541787811],
    [-10.682634879184775, -1.01393349473053, 0.05975540930144163],
    [13.006982934478081, -11.259112365356977, 3.7143339235154684],
    [-2.5738583162520285, -6.851549044569985, -7.581618914036472],
    [-19.92867823877704, 7.0518316214987635, 7.33055142817776],
]