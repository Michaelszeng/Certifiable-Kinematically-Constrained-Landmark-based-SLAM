import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize_results(N, K, t, v, R, p, Omega, z=None, log=True):
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
        
    # Convert np arrays to lists if needed
    if isinstance(t, np.ndarray):
        t_new = [element for element in t]
    else:
        t_new = t
    if isinstance(v, np.ndarray):
        v_new = [element for element in v]
    else:
        v_new = v
    if isinstance(R, np.ndarray):
        R_new = [element for element in R]
    else:
        R_new = R
    if isinstance(p, np.ndarray):
        p_new = [element for element in p]
    else:
        p_new = p
    if isinstance(Omega, np.ndarray):
        Omega_new = [element for element in Omega]
    else:
        Omega_new = Omega
        
    if _2d: 
        visualize_results_2D(N, K, t_new, v_new, R_new, p_new, Omega_new, log)
    elif z is not None:
        visualize_results_3D_moving_landmarks(N, K, t_new, v_new, R_new, p_new, Omega_new, z, log)
    else:
        visualize_results_3D(N, K, t_new, v_new, R_new, p_new, Omega_new, log)
        

def visualize_results_2D(N, K, t, v, R, p, Omega, log=True):
    """
    Plots the robot position and orientation, velocity, and landmark positions
    on a 2D plane to help visualize optimization results.
    """
    if log:
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
            rotation_matrix: rotation matrix
            size: Size of the triangle
            color: Color of the triangle
        """
        # Define a basic triangle pointing up (relative to local frame)
        triangle = np.array([[size, 0], [-size/2, size/2], [-size/2, -size/2]])  # X is forward
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
    
    
def visualize_results_3D(N, K, t, v, R, p, Omega, log=True, animate=False, gridlines=True):
    """
    Plots the robot position and orientation, velocity, and landmark positions
    in 3D to help visualize optimization results.
    """
    if log:
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
            [size, 0, 0], 
            [-size/2, size/2, 0], 
            [-size/2, -size/2, 0]
        ])  # X is forward
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
    
    if not gridlines:
        ax.grid(False)
        ax.axis('off')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Robot Poses and Landmarks')
        ax.legend()
    
    # Animation function to update the view angle
    def update(frame):
        ax.view_init(elev=30, azim=frame)

    # Create animation
    if animate:
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
    plt.show()
    
    
def visualize_results_3D_moving_landmarks(N, K, t, v, R, p, Omega, z, log=True, animate=True, gridlines=True):
    """
    Plots the robot position and orientation, velocity, and moving landmark positions
    in 3D to help visualize optimization results.
    Args:
        N: Number of robot poses
        K: Number of landmarks
        t: List of robot positions (Nx3 numpy array)
        v: List of robot velocities (Nx3 numpy array)
        R: List of robot orientations as 3x3 rotation matrices (Nx3x3 numpy array)
        p: List of initial landmark positions (Kx3 numpy array)
        Omega: Optional data (ignored here for simplicity)
        z: List of constant velocities for each landmark (Kx3 numpy array)
    """
    if log:
        print(f"t: \n{np.array(t)}")
        print(f"v: \n{np.array(v)}")
        print(f"p: \n{np.array(p)}")
        print(f"R: \n{np.array(R)}")
        print(f"Omega: \n{np.array(Omega)}")
        print(f"z: \n{np.array(z)}")

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
            [size, 0, 0], 
            [-size / 2, size / 2, 0], 
            [-size / 2, -size / 2, 0]
        ])  # X is forward
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
    for i in range(N - 1):
        position = np.array(t[i])
        velocity = R[i] @ v[i]

        # Draw velocity vector
        ax.quiver(
            position[0], position[1], position[2],
            velocity[0], velocity[1], velocity[2],
            color='green', length=np.linalg.norm(velocity), normalize=True, arrow_length_ratio=0.1
        )

    # Plot moving landmarks and their paths
    for k in range(K):
        landmark_positions = [p[k] + z[k] * i for i in range(N)]
        landmark_positions = np.array(landmark_positions)

        # Plot path of the landmark
        ax.plot(
            landmark_positions[:, 0], landmark_positions[:, 1], landmark_positions[:, 2],
            color='red', alpha=0.5, linestyle='--', label=f'Landmark {k + 1} Path' if k == 0 else None
        )

        # Plot the landmark positions over time with translucency
        for i in range(N):
            alpha = 0.2 + 0.8 * (i / (N - 1))  # Older points are more translucent
            ax.scatter(
                landmark_positions[i, 0], landmark_positions[i, 1], landmark_positions[i, 2],
                color='red', alpha=alpha, s=50
            )

    # Set equal axis scaling
    all_points = np.vstack([np.array(t)] + [p + z * i for i in range(N)])
    x_limits = [all_points[:, 0].min(), all_points[:, 0].max()]
    y_limits = [all_points[:, 1].min(), all_points[:, 1].max()]
    z_limits = [all_points[:, 2].min(), all_points[:, 2].max()]
    max_range = np.array([
        x_limits[1] - x_limits[0], 
        y_limits[1] - y_limits[0], 
        z_limits[1] - z_limits[0]
    ]).max() / 2.0

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if not gridlines:
        ax.grid(False)
        ax.axis('off')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Robot Poses and Moving Landmarks')
        ax.legend()

    # Animation function to update the view angle
    def update(frame):
        ax.view_init(elev=30, azim=frame)

    # Create animation
    if animate:
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
    plt.show()
