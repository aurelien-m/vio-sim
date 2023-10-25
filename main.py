import rerun as rr

from robot import Robot
from numpy.linalg import norm
import numpy as np
from utils import normalize


def log_frame(position: np.array, R: np.array) -> rr.Arrows3D:
    x = normalize(R @ np.array([1, 0, 0]))
    y = normalize(R @ np.array([0, 1, 0]))
    z = normalize(R @ np.array([0, 0, 1]))

    origins = [position, position, position]
    vectors = [x, y, z]
    rr.log(
        "world/frame",
        rr.Arrows3D(
            origins=origins,
            vectors=vectors,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

    return origins, vectors


if __name__ == "__main__":
    trajectory = [[0, 0, 20], [13, 8, 21], [22, 12, 23], [27, 24, 21]]
    robot = Robot(trajectory)
    rr.init("robot simulator", spawn=True)

    positions = []
    frame_origins = []
    frame_vectors = []
    frame_colors = []

    while robot.moving:
        robot.step()

        rr.set_time_seconds("robot_clock", robot.clock)
        rr.log("robot/velocity_norm", rr.Tensor([norm(robot.velocity)]))
        rr.log("robot/velocity", rr.Tensor(robot.velocity))
        rr.log("robot/acceleration_norm", rr.Tensor([norm(robot.acceleration)]))
        rr.log("robot/acceleration", rr.Tensor(robot.acceleration))
        rr.log("robot/angle_velocity", rr.Tensor(robot.angle_velocity))
        rr.log("world/position", rr.Points3D(robot.position))

        origins, vectors = log_frame(robot.position, robot.R.T)
        frame_origins += origins
        frame_vectors += vectors
        frame_colors += [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

        positions.append(robot.position)

    rr.log("world/trajectory", rr.Points3D(positions), timeless=True)
    rr.log(
        "world/frames",
        rr.Arrows3D(origins=frame_origins, vectors=frame_vectors, colors=frame_colors),
        timeless=True,
    )
