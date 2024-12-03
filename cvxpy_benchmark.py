import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from certifiable_solver import certifiable_solver
from solver_utils import *

true_lin_vel = np.array([1, 0, 0.5])
true_rpy_vel = np.array([0, 0, 45])

num_landmarks = 4
num_timesteps = 4
samples_per_noise = 10

noise_levels = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0]

average_ranks = []

for noise in noise_levels:
    ranks = []
    print(f"Noise: {noise}\n")
    for _ in range(samples_per_noise):
        true_landmarks = np.random.uniform(-10, 10, size=(num_landmarks, 3))
        true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

        true_lin_pos, true_ang_pos = generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel)
        print_ground_truth(true_ang_vel, true_ang_pos, true_landmarks, true_lin_vel, true_lin_pos)

        measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=noise, dropout=0.0)
        
        calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S = certifiable_solver(measurements)
        print_results(calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S)
        ranks.append(rank)
    
    average_ranks.append(np.mean(ranks))

average_ranks = np.array(average_ranks)


plt.figure(figsize=(10, 6))
positions = np.arange(len(noise_levels))
plt.bar(positions, average_ranks, width=0.9, color='b', alpha=0.7)
plt.xticks(positions, labels=noise_levels, rotation=45)
plt.xlabel("Noise Level", fontsize=14)
plt.ylabel("Average Rank", fontsize=14)
plt.title("Average Rank vs. Noise Level", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

