import numpy as np
import matplotlib.pyplot as plt

# Load the two benchmarks for landmarks and timesteps
file1_lm = "spiral_small_times_lm.npy"
file2_lm = "straight_small_times_lm.npy"
file1_t = "spiral_small_times_t.npy"
file2_t = "straight_small_times_t.npy"

times_lm_1 = np.load(file1_lm)
times_lm_2 = np.load(file2_lm)
times_t_1 = np.load(file1_t)
times_t_2 = np.load(file2_t)

# Number of landmarks and timesteps
num_landmarks = [4, 8, 12, 16, 20, 24, 28]
num_timesteps = [3, 5, 6, 7, 8, 9]

# Plot settings
width = 0.4  # Bar width

# Landmarks Timing Graph
plt.figure(figsize=(10, 6))
positions = np.arange(len(num_landmarks))
plt.bar(positions - width / 2, times_lm_1, width, label="Spiral", color='b', alpha=0.7)
plt.bar(positions + width / 2, times_lm_2, width, label="Line", color='g', alpha=0.7)
plt.xticks(positions, labels=num_landmarks, rotation=45)
plt.xlabel("Number of Landmarks", fontsize=14)
plt.ylabel("Average Time (s)", fontsize=14)
plt.title("Average Time vs. Number of Landmarks", fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("combined_times_lm_plot.png")
plt.show()

# Timesteps Timing Graph
plt.figure(figsize=(10, 6))
positions = np.arange(len(num_timesteps))
plt.bar(positions - width / 2, times_t_1, width, label="Spiral", color='b', alpha=0.7)
plt.bar(positions + width / 2, times_t_2, width, label="Line", color='g', alpha=0.7)
plt.xticks(positions, labels=num_timesteps, rotation=45)
plt.xlabel("Number of Timesteps", fontsize=14)
plt.ylabel("Average Time (s)", fontsize=14)
plt.title("Average Time vs. Number of Timesteps", fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("combined_times_t_plot.png")
plt.show()
