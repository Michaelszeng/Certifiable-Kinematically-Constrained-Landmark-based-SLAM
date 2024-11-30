import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def visualize_results(N, K, t, v, R, p):
    """
    Plots the robot position and orientation, velocity, and landmark positions
    on a 2D plane to help visualize optimization results.
    
    Args:
        N: number of time stmps
        K: number of landmarks
        t: list of robot translations (1D np arrays)
        v: list of robot velocities (2D np arrays)
        R: list of robot rotation matrices (2D np arrays)
        p: list of landmark positions (1D np arrays)
    """
    def draw_triangle(ax, position, rotation_matrix, size=0.3, color='blue'):
        """
        Draw a triangle to represent the robot pose.
        Args:
            ax: Matplotlib axis
            position: 2D position of the robot (x, y)
            rotation_matrix: 2x2 rotation matrix
            size: Size of the triangle
            color: Color of the triangle
        """
        # Define a basic triangle pointing up (relative to local frame)
        triangle = np.array([[0, size], [size/2, -size/2], [-size/2, -size/2]])
        # Rotate and translate triangle
        rotated_triangle = (rotation_matrix @ triangle.T).T + position
        polygon = Polygon(rotated_triangle, closed=True, color=color)
        ax.add_patch(polygon)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Plot each robot pose as a triangle
    for i in range(N):
        position = np.array([t[i][0], t[i][1]])  # Ignore z-axis
        rotation_matrix = R[i][:2, :2]  # Use the top-left 2x2 of the 3x3 matrix
        
        # Draw robot pose
        draw_triangle(ax, position, rotation_matrix, size=0.3, color='blue')
    
    # Plot each velocity as an arrow
    for i in range(N-1):
        position = np.array([t[i][0], t[i][1]])  # Ignore z-axis
        v_i_rotated = R[i] @ v[i]
        velocity = np.array([v_i_rotated[0], v_i_rotated[1]])  # Ignore z-axis
        
        # Draw velocity vector
        ax.quiver(
            position[0], position[1],
            velocity[0], velocity[1],
            angles='xy', scale_units='xy', scale=1, color='green'
        )

    # Plot each point p[k] in red
    for k in range(K):
        point = np.array([p[k][0], p[k][1]])  # Ignore z-axis
        ax.plot(point[0], point[1], 'ro')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Visualization of Robot Poses and Points')

    plt.grid()
    plt.show()