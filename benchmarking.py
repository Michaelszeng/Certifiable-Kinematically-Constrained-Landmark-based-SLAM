import numpy as np
import time
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from certifiable_solver import certifiable_solver
from solver_utils import *
from evaluation import compute_relaxation_gap, compute_mean_errors

# TRIAL = "LINE_N=4_K=4"
# TRIAL = "SPIRAL_N=4_K=4"
TRIAL = "LINE_N=8_K=8"
# TRIAL = "SPIRAL_N=8_K=8"

print(f"{TRIAL}")
time.sleep(3)

if TRIAL == "LINE_N=4_K=4":
    true_lin_vel = np.array([1, 0, 0])
    true_rpy_vel = np.array([0, 0, 0])
    num_landmarks = 4
    num_timesteps = 4
elif TRIAL == "SPIRAL_N=4_K=4":
    true_lin_vel = np.array([1, 0, 0.5])
    true_rpy_vel = np.array([0, 0, 45])
    num_landmarks = 4
    num_timesteps = 4
elif TRIAL == "LINE_N=8_K=8":
    true_lin_vel = np.array([1, 0, 0])
    true_rpy_vel = np.array([0, 0, 0])
    num_landmarks = 8
    num_timesteps = 8
elif TRIAL == "SPIRAL_N=8_K=8":
    true_lin_vel = np.array([1, 0, 0.5])
    true_rpy_vel = np.array([0, 0, 45])
    num_landmarks = 8
    num_timesteps = 8
    
    
samples_per_noise = 10

noise_levels = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0]

averages = {
    "gap": [],
    "t": [],
    "v": [],
    "R": [],
    "p": [],
    "Omega": [],
}
average_ranks = []

for noise in noise_levels:
    curr_averages = {
        "gap": [],
        "t": [],
        "v": [],
        "R": [],
        "p": [],
        "Omega": [],
    }
    ranks = []
    for _ in range(samples_per_noise):
        true_landmarks = np.random.uniform(-10, 10, size=(num_landmarks, 3))
        true_ang_vel = Rotation.from_euler("xyz", true_rpy_vel, degrees=True).as_matrix()

        true_lin_pos, true_ang_pos = generate_ground_truth(num_timesteps, true_lin_vel, true_ang_vel)
        print_ground_truth(true_ang_vel, true_ang_pos, true_landmarks, true_lin_vel, true_lin_pos)

        measurements = generate_measurements(true_lin_pos, true_ang_pos, true_landmarks, noise=noise)
        
        print(f"\nBEGINNING NOISE LEVEL {noise} TRIAL {_}\n")
        calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S = certifiable_solver(measurements)
        print_results(calc_ang_vel, calc_ang_pos, calc_landmarks, calc_lin_vel, calc_lin_pos, rank, S)

        gap = compute_relaxation_gap(measurements, calc_lin_pos, calc_lin_vel, calc_ang_pos, calc_landmarks, calc_ang_vel, 
                                     true_lin_pos, true_lin_vel, true_ang_pos, true_landmarks, true_ang_vel)
        curr_averages["gap"].append(gap)

        mean_errors = compute_mean_errors(measurements, calc_lin_pos, calc_lin_vel, calc_ang_pos, calc_landmarks, calc_ang_vel, 
                                                        true_lin_pos, true_lin_vel, true_ang_pos, true_landmarks, true_ang_vel)    
        for key, val in mean_errors.items():
            curr_averages[key].append(val)
            
        ranks.append(rank)

    for key, val in curr_averages.items():
        averages[key].append(float(np.mean(val)))
        
    average_ranks.append(np.mean(ranks))

average_ranks = np.array(average_ranks)

np.save(f"error_benchmark_{TRIAL}.npy", averages)
np.save(f"rank_benchmark_{TRIAL}.npy", average_ranks)

# Errors plots
texts = {
    "gap": "Relaxation Gap",
    "t": "Translation Error",
    "R": "Rotation Error",
    "p": "Landmark Error",
}
for key, text in texts.items():
    plt.figure(figsize=(5, 3.2))
    positions = np.arange(len(noise_levels))
    plt.bar(positions, averages[key], width=0.9, color='b', alpha=0.7)
    plt.xticks(positions, labels=noise_levels, rotation=45)
    plt.xlabel("Noise Level", fontsize=14)
    plt.ylabel(f"Average {text}", fontsize=14)
    plt.title(f"{text} vs. Noise Level", fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Save as an image file
    plt.savefig(f"benchmark_{key}_{TRIAL}.png", dpi=300)  # Save as PNG


# Rank plot
plt.figure(figsize=(5, 3.2))
positions = np.arange(len(noise_levels))
plt.bar(positions, average_ranks, width=0.9, color='b', alpha=0.7)
plt.xticks(positions, labels=noise_levels, rotation=45)
plt.xlabel("Noise Level", fontsize=14)
plt.ylabel("Average Rank", fontsize=14)
plt.title("Average Rank vs. Noise Level", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Save as an image file
plt.savefig(f"benchmark_rank_{TRIAL}.png", dpi=300)  # Save as PNG