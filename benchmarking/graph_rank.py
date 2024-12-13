import numpy as np
import matplotlib.pyplot as plt

# Define file names
files = [
    "spiral_small_rank.npy",
    "line_small_rank.npy",
    "spiral_large_rank.npy",
    "line_large_rank.npy",
    "curve_rank.npy",
]

# Load the .npy files
averages = []
for file in files:
    averages.append(np.load(file))

# Define the noise levels used in the experiment
noise_levels = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0]
positions = np.arange(len(noise_levels))  # Bar positions

# Create side-by-side bars
margin = 0.2
width = (1 - margin) / len(files)
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'gray']
plt.figure(figsize=(10, 6))
for i, file in enumerate(files):
    label = file[:-4]
    color = colors[i % len(colors)]
    plt.bar(positions - 0.5 + (margin + width) / 2 + i * width, averages[i], width, label=label, color=color, alpha=0.7)

# Customizing the plot
plt.xticks(positions, labels=noise_levels, rotation=45)
plt.xlabel("Noise Level", fontsize=14)
plt.ylabel("Average Rank", fontsize=14)
plt.title("Average Rank vs. Noise Level", fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as an image
plt.savefig("output_files/rank_comparison.png")

# Show the plot
plt.show()
