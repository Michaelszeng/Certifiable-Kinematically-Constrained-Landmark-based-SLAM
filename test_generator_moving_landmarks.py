import numpy as np
import time
from scipy.spatial.transform import Rotation   
from solver_cvxpy import certifiable_solver
from solver_utils import *
from visualization_utils import visualize_results
from evaluation import compute_relaxation_gap, compute_mean_errors

true_lin_vel = np.array([1, 0, 0.5])
true_rpy_vel = np.array([0, 0, 45])

num_landmarks = 8
num_timesteps = 8

# Velocity, angular velocity, and measurement covariances
cov_v = 1
cov_omega = 1
cov_meas = 1

# Define ranges for each dimension
x_range = (-5, 5)
y_range = (-5, 5)
z_range = (-5, 5)

x_coords = np.random.uniform(x_range[0], x_range[1], size=num_landmarks)
y_coords = np.random.uniform(y_range[0], y_range[1], size=num_landmarks)
z_coords = np.random.uniform(z_range[0], z_range[1], size=num_landmarks)
true_landmarks = np.stack((x_coords, y_coords, z_coords), axis=1)

true_landmark_vel = np.random.uniform(-0.5, 0.5, size=(num_landmarks, 3))
true_landmark_vel = 0.25*(true_landmark_vel / np.linalg.norm(true_landmark_vel, axis=1, keepdims=True))  # all velocities are 0.25 in magnitude

true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

true_lin_pos, true_ang_pos = generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel)
print_ground_truth_moving_landmarks(true_ang_vel, true_ang_pos, true_landmarks, true_lin_vel, true_lin_pos, true_landmark_vel)

measurements = generate_measurements_moving_landmarks(true_lin_pos, true_ang_pos, true_landmarks, true_landmark_vel, noise=0.0, dropout=0.0)

# Generate a new `testX.py` file to save this test case
generate_test_file_moving_landmarks("test_data/test_moving_landmarks_spiral.py", measurements, true_lin_pos, true_lin_vel, true_landmarks, true_ang_pos, true_ang_vel, true_landmark_vel)