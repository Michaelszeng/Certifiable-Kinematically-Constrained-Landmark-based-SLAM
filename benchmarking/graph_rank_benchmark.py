import numpy as np
import matplotlib.pyplot as plt

# Define file names
file1 = "spiral_small_rank.npy"
file2 = "straight_small_rank.npy"

# Load the two .npy files
average_ranks1 = np.load(file1)
average_ranks2 = np.load(file2)

# Define the noise levels used in the experiment
noise_levels = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0]
positions = np.arange(len(noise_levels))  # Bar positions

# Plotting the side-by-side bar graph
width = 0.4  # Bar width
plt.figure(figsize=(10, 6))

plt.bar(positions - width / 2, average_ranks1, width, label="Spiral", color='b', alpha=0.7)
plt.bar(positions + width / 2, average_ranks2, width, label="Line", color='g', alpha=0.7)

# Customizing the plot
plt.xticks(positions, labels=noise_levels, rotation=45)
plt.xlabel("Noise Level", fontsize=14)
plt.ylabel("Average Rank", fontsize=14)
plt.title("Average Rank vs. Noise Level", fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as an image
plt.savefig("combined_rank_plot.png")

# Show the plot
plt.show()
