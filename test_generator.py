import numpy as np
from scipy.spatial.transform import Rotation   
from certifiable_solver import certifiable_solver
from solver_utils import generate_ground_truth, generate_measurements, print_results

# TODO: use yaml to input these things
true_lin_vel = np.array([2, 0, 0])
true_rpy_vel = np.array([22.5, 0, 0])
true_landmarks = np.array([
    [-1, -1, 1],
    [3, 3, -1],
    [-1, 2, -2],
    [1, -10, 2],
    [-3, 2, 12],
    [0, 4, 1],
    [-3, 1, 5],
    [3, 3, -3],
    [-1, 2, -12],
    [10, -10, 2],
    [-13, 0, 2],
    [30, 14, 1],
    [60, -4, 12],
    [-13, 3, 4],
    [15, 15, -30],
    [-8, 12, -4],
    [5, -8, 2],
    [-2, 2, 2],
    [13, 19, -12],
])
num_landmarks = 30
num_timesteps = 3

true_landmarks = np.random.uniform(-100, 100, size=(num_landmarks, 3))
true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

true_lin_pos, true_ang_pos = generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel)
measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks)
calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank = certifiable_solver(measurements)
print_results(calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank)

