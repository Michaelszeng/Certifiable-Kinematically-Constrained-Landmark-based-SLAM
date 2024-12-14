import numpy as np

d = 3   # dimension of space (3D)

# Measurement data: maps landmark to {timestamp: measurement} dicts
y_bar = {
    0: {
        0: np.array([[-1.3759,4.3791,-4.8246]]).T,
        1: np.array([[-2.5072,4.2084,-4.9516]]).T,
        2: np.array([[-3.6385,4.0377,-5.0787]]).T,
        3: np.array([[-4.7698,3.8671,-5.2057]]).T,
    },
    1: {
        0: np.array([[3.6499,3.0299,-0.1136]]).T,
        1: np.array([[2.7972,2.9935,-0.3124]]).T,
        2: np.array([[1.9445,2.9572,-0.5111]]).T,
        3: np.array([[1.0918,2.9208,-0.7098]]).T,
    },
    2: {
        0: np.array([[-2.8927,2.2431,-4.1810]]).T,
        1: np.array([[-3.6575,2.2094,-4.1035]]).T,
        2: np.array([[-4.4222,2.1757,-4.0259]]).T,
        3: np.array([[-5.1870,2.1419,-3.9483]]).T,
    },
    3: {
        0: np.array([[-0.2731,0.9416,3.3824]]).T,
        1: np.array([[-1.1816,1.0709,3.1890]]).T,
        2: np.array([[-2.0901,1.2002,2.9956]]).T,
        3: np.array([[-2.9987,1.3295,2.8021]]).T,
    },
    4: {
        0: np.array([[2.8481,1.7646,1.3392]]).T,
        1: np.array([[1.8229,1.6730,1.1080]]).T,
        2: np.array([[0.7978,1.5814,0.8767]]).T,
        3: np.array([[-0.2273,1.4898,0.6454]]).T,
    },
}

N = 1
for lm_meas in y_bar.values():
    for timestep in lm_meas.keys():
        N = max(N, timestep + 1)
K = len(y_bar)

# Covariances
cov_v=1
cov_omega=1
cov_meas=1

# Ground Truths
t_gt = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [3.0, 0.0, 0.0],
])
R_gt = np.array([
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
])
v_gt = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
])
Omega_gt = np.array([
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
])
p_gt = np.array([
    [-1.3759114409504916, 4.379050273436805, -4.824605254006949],
    [3.6499454423068514, 3.0298791207138525, -0.1136365728538653],
    [-2.8927193964544817, 2.2431112289983552, -4.1810420908160895],
    [-0.2730857024484621, 0.941626743615763, 3.3824057103918914],
    [2.84806648460607, 1.7645837672995643, 1.339222793063941],
])
z_gt = np.array([
    [-0.13130706605756293, -0.17065558461404406, -0.12702411520413615],
    [0.14727291382512578, -0.036360780062154055, -0.1987173432962728],
    [0.23524960498304215, -0.03372181538734853, 0.07759163951292826],
    [0.09147677387166564, 0.12930723801996477, -0.19342088314781403],
    [-0.02512743757777064, -0.09158565457076764, -0.23125890200729413],
])