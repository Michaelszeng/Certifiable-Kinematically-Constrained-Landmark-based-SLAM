import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def visualize_results(N, K, t, v, R, p):
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

    # Plot each robot pose as a triangle and velocity as an arrow
    for i in range(N):
        position = np.array([t[i][0], t[i][1]])  # Ignore z-axis
        rotation_matrix = R[i][:2, :2]  # Use the top-left 2x2 of the 3x3 matrix
        v_i_rotated = R[i] @ v[i]
        velocity = np.array([v_i_rotated[0], v_i_rotated[1]])  # Ignore z-axis
        
        # Draw robot pose
        draw_triangle(ax, position, rotation_matrix, size=0.3, color='blue')
        
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

def visualize_results_3d(calc_landmarks, calc_lin_pos):
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot landmarks
    ax.scatter(calc_landmarks[:, 0], calc_landmarks[:, 1], calc_landmarks[:, 2], color='r', label='Landmarks')

    # Plot linear position
    ax.plot(calc_lin_pos[:, 0], calc_lin_pos[:, 1], calc_lin_pos[:, 2], marker='o', label='Linear Position')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Landmarks and Positions')

    # Set the limits for the axes to be the same
    max_range = np.ptp(np.concatenate([calc_landmarks, calc_lin_pos], axis=0), axis=0).max()
    mid_point = np.mean(np.concatenate([calc_landmarks, calc_lin_pos], axis=0), axis=0)

    # Set equal scaling by adjusting limits
    ax.set_xlim(mid_point[0] - max_range, mid_point[0] + max_range)
    ax.set_ylim(mid_point[1] - max_range, mid_point[1] + max_range)
    ax.set_zlim(mid_point[2] - max_range, mid_point[2] + max_range)

    # Add legend
    ax.legend()

    # Show plot
    plt.show()
