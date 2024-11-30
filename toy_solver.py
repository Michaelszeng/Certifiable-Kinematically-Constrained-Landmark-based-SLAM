import numpy as np
from certifiable_solver import certifiable_solver
from solver_utils import print_results

# Outer label: landmark number
# Inner label: timestep
measurements = {
    # Landmark number
    0: {
        # Timestep number
        1: np.array([[6,-3,1]]).T,
        2: np.array([[2,-3,1]]).T,
        3: np.array([[-2,-3,1]]).T,
    },
    1: {
        2: np.array([[0,3,-1]]).T,
        3: np.array([[-4,3,-1]]).T,
    },
    2: {
        0: np.array([[9,7,-4]]).T,
        1: np.array([[5,7,-4]]).T,
        3: np.array([[-3,7,-4]]).T,
    },
    3: {
        0: np.array([[1,2,2]]).T,
        2: np.array([[-7,2,2]]).T,
        3: np.array([[-11,2,2]]).T,
    },
    4: {
        0: np.array([[0,4,1]]).T,
        1: np.array([[-4,4,1]]).T,
        2: np.array([[-8,4,1]]).T,
    },
}

ang_vel, ang_pos, landmarks, lin_vel, lin_pos, rank, S = certifiable_solver(measurements, tol=0.01)
print_results(ang_vel, ang_pos, landmarks, lin_vel, lin_pos, rank, S)
