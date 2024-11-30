import numpy as np
from scipy.spatial.transform import Rotation   
from certifiable_solver import certifiable_solver
from solver_utils import generate_ground_truth, generate_measurements, print_ground_truth, print_results

true_lin_vel = np.array([2, 0, 1])
true_rpy_vel = np.array([20, 0, 0])

num_landmarks = 3
num_timesteps = 5

true_landmarks = np.random.uniform(-50, 50, size=(num_landmarks, 3))
true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

true_lin_pos, true_ang_pos = generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel)
print_ground_truth(true_ang_vel, true_ang_pos, true_landmarks, true_lin_vel, true_lin_pos)


measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks)
calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S = certifiable_solver(measurements)
print_results(calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S)

