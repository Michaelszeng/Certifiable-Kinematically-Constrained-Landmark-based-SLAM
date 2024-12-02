import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize_results(N, K, t, v, R, p, Omega):
    """
    Check if data is 2D or 3D and use appropriate plotting function.
    
    Args:
        N: number of time stmps
        K: number of landmarks
        t: list of robot translations (1D np arrays)
        v: list of robot velocities (2D np arrays)
        R: list of robot rotation matrices (2D np arrays)
        p: list of landmark positions (1D np arrays)
        Omega: list of robot angular velocity matrices (2D np arrays) (just for printing -- not used in visual)
    """
    _2d = True
    for landmark in p:
        if landmark[2] != 0:
            _2d = False
            break
        
    if _2d: 
        visualize_results_2D(N, K, t, v, R, p, Omega)
    else:
        visualize_results_3D(N, K, t, v, R, p, Omega)
        

def visualize_results_2D(N, K, t, v, R, p, Omega):
    """
    Plots the robot position and orientation, velocity, and landmark positions
    on a 2D plane to help visualize optimization results.
    """
    print(f"t: \n{np.array(t)}")
    print(f"v: \n{np.array(v)}")
    print(f"p: \n{np.array(p)}")
    print(f"R: \n{np.array(R)}")
    print(f"Omega: \n{np.array(Omega)}")
    
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
    ax.set_title('2D Visualization of Robot Poses and Landmarks')

    plt.grid()
    plt.show()
    
    
def visualize_results_3D(N, K, t, v, R, p, Omega):
    """
    Plots the robot position and orientation, velocity, and landmark positions
    in 3D to help visualize optimization results.
    """
    print(f"t: \n{np.array(t)}")
    print(f"v: \n{np.array(v)}")
    print(f"p: \n{np.array(p)}")
    print(f"R: \n{np.array(R)}")
    print(f"Omega: \n{np.array(Omega)}")
    
    def draw_triangle(ax, position, rotation_matrix, size=0.3, color='blue'):
        """
        Draw a triangle to represent the robot pose in 3D.
        Args:
            ax: Matplotlib 3D axis
            position: 3D position of the robot (x, y, z)
            rotation_matrix: 3x3 rotation matrix
            size: Size of the triangle
            color: Color of the triangle
        """
        # Define a basic triangle pointing up (relative to local frame) in 3D
        triangle = np.array([
            [0, size, 0], 
            [size/2, -size/2, 0], 
            [-size/2, -size/2, 0]
        ])
        # Rotate and translate triangle
        rotated_triangle = (rotation_matrix @ triangle.T).T + position
        # Create a 3D polygon
        polygon = Poly3DCollection([rotated_triangle], color=color, alpha=0.5)
        ax.add_collection3d(polygon)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each robot pose as a triangle
    for i in range(N):
        position = np.array(t[i]) 
        rotation_matrix = R[i]
        
        # Draw robot pose
        draw_triangle(ax, position, rotation_matrix, size=0.3, color='blue')
    
    # Plot each velocity as an arrow
    for i in range(N-1):
        position = np.array(t[i])
        velocity = R[i] @ v[i]
        
        # Draw velocity vector
        ax.quiver(
            position[0], position[1], position[2], 
            velocity[0], velocity[1], velocity[2], 
            color='green', length=np.linalg.norm(velocity), normalize=True, arrow_length_ratio=0.1
        )

    # Plot each landmark in red
    for k in range(K):
        point = np.array(p[k])
        ax.scatter(point[0], point[1], point[2], color='red', s=50, label='Landmark' if k == 0 else "")
        
    # Set equal axis scaling
    all_points = np.array(t + p)
    x_limits = [all_points[:, 0].min(), all_points[:, 0].max()]
    y_limits = [all_points[:, 1].min(), all_points[:, 1].max()]
    z_limits = [all_points[:, 2].min(), all_points[:, 2].max()]
    max_range = np.array([x_limits[1] - x_limits[0], 
                          y_limits[1] - y_limits[0], 
                          z_limits[1] - z_limits[0]]).max() / 2.0

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Robot Poses and Landmarks')
    ax.legend()
    plt.show()