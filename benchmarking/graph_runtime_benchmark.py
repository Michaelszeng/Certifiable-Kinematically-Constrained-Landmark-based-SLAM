import numpy as np
import matplotlib.pyplot as plt

# Load the benchmarks for landmarks
landmark_files = [
    "spiral_small_times_lm.npy",
    "line_small_times_lm.npy",
]

# Load the benchmarks for timesteps
timestep_files = [
    "spiral_small_times_t.npy",
    "line_small_times_t.npy",
]

# Load the .npy files
averages_lm = []
averages_t = []
for file in landmark_files:
    averages_lm.append(np.load(file))
for file in timestep_files:
    averages_t.append(np.load(file))

# Number of landmarks and timesteps
num_landmarks = [4, 8, 12, 16, 20, 24, 28]
num_timesteps = [3, 5, 6, 7, 8, 9]

# Create side-by-side bars
positions = np.arange(len(num_landmarks))
width = (1 - 0.2) / len(landmark_files)
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'gray']
plt.figure(figsize=(10, 6))
for i, file in enumerate(landmark_files):
    label = file[:-4]
    color = colors[i % len(colors)]
    plt.bar(positions - 0.4 + width / 2 + i * width, averages_lm[i], width, label=label, color=color, alpha=0.7)

# Landmarks Timing Graph
plt.xticks(positions, labels=num_landmarks, rotation=45)
plt.xlabel("Number of Landmarks", fontsize=14)
plt.ylabel("Average Time (s)", fontsize=14)
plt.title("Average Time vs. Number of Landmarks", fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("combined_times_lm_plot.png")
plt.show()

# Create side-by-side bars
positions = np.arange(len(num_timesteps))
width = (1 - 0.2) / len(timestep_files)
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'gray']
plt.figure(figsize=(10, 6))
for i, file in enumerate(timestep_files):
    label = file[:-4]
    color = colors[i % len(colors)]
    plt.bar(positions - 0.4 + width / 2 + i * width, averages_t[i], width, label=label, color=color, alpha=0.7)

# Timesteps Timing Graph
plt.xticks(positions, labels=num_timesteps, rotation=45)
plt.xlabel("Number of Timesteps", fontsize=14)
plt.ylabel("Average Time (s)", fontsize=14)
plt.title("Average Time vs. Number of Timesteps", fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("combined_times_t_plot.png")
plt.show()
