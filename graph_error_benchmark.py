import numpy as np
import matplotlib.pyplot as plt

# Define file names
file1 = "spiral_small_error.npy"
file2 = "straight_small_error.npy"
file3 = "spiral_large_error.npy"
file4 = "straight_large_error.npy"

# Load the two .npy files
averages1 = np.load(file1, allow_pickle=True).item()
averages2 = np.load(file2, allow_pickle=True).item()
averages3 = np.load(file3, allow_pickle=True).item()
averages4 = np.load(file4, allow_pickle=True).item()

# Define the noise levels used in the experiment
noise_levels = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0]
positions = np.arange(len(noise_levels))  # Bar positions

# Plot settings for each metric
texts = {
    "gap": "Relaxation Gap",
    "t": "Translation Error",
    "R": "Rotation Error",
    "p": "Landmark Error",
}
width = 0.2  # Bar width

# Plot each metric
for key, text in texts.items():
    plt.figure(figsize=(10, 6))
    
    # Create side-by-side bars for Dataset 1 and Dataset 2
    plt.bar(positions - 3*width / 2, averages1[key], width, label="spiral_small", color='r', alpha=0.7)
    plt.bar(positions - width / 2, averages2[key], width, label="line_small", color='g', alpha=0.7)
    plt.bar(positions + width / 2, averages3[key], width, label="spiral_large", color='b', alpha=0.7)
    plt.bar(positions + 3*width / 2, averages4[key], width, label="line_large", color='c', alpha=0.7)
    
    # Customize the plot
    plt.xticks(positions, labels=noise_levels, rotation=45)
    plt.xlabel("Noise Level", fontsize=14)
    plt.ylabel(f"Average {text}", fontsize=14)
    plt.title(f"{text} vs. Noise Level", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot as an image
    name = text.lower().replace(" ", "")
    plt.savefig(f"plots/{name}_plot.png")
    
    # Show the plot
    plt.show()
