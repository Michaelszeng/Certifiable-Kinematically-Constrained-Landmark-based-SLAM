import numpy as np
import matplotlib.pyplot as plt

# Define file names
files = [
    "spiral_small_error.npy",
    "line_small_error.npy",
    #"spiral_large_error.npy",
    #"line_large_error.npy",
    "curve_error.npy",
]

# Load the .npy files
averages = []
for file in files:
    averages.append(np.load(file, allow_pickle=True).item())

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

# Plot each metric
for key, text in texts.items():
    plt.figure(figsize=(10, 6))
    
    # Create side-by-side bars
    margin = 0.2
    width = (1 - margin) / len(files)
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'gray']
    for i, file in enumerate(files):
        label = file[:-4]
        color = colors[i % len(colors)]
        plt.bar(positions - 0.5 + (margin + width) / 2 + i * width, averages[i][key], width, label=label, color=color, alpha=0.7)
    
    # Customize the plot
    plt.xticks(positions, labels=noise_levels, rotation=45)
    plt.xlabel("Noise Level", fontsize=14)
    plt.ylabel(f"Average {text}", fontsize=14)
    plt.title(f"{text} vs. Noise Level", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(f"{key}_error_comparison.png")
    
    # Show the plot
    plt.show()
