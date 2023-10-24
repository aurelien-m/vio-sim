import rerun as rr

from robot import Robot
from numpy.linalg import norm
import numpy as np


def log_frame(position: np.array, R: np.array) -> rr.Arrows3D:
    x = R @ np.array([1, 0, 0])
    y = R @ np.array([0, 1, 0])
    z = R @ np.array([0, 0, 1])

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
    trajectory = [[0, 0, 0], [13, 8, 1], [22, 12, 3], [27, 24, 9]]
    robot = Robot(trajectory)
    rr.init("robot simulator", spawn=True)

    positions = []
    frame_origins = []
    frame_vectors = []
    frame_colors = []

    while robot.moving:
        robot.step()

        rr.set_time_seconds("robot_clock", robot.clock)
        rr.log("robot/velocity", rr.Tensor([norm(robot.velocity)]))
        rr.log("robot/acceleration", rr.Tensor([norm(robot.acceleration)]))
        rr.log("world/position", rr.Points3D(robot.position))

        origins, vectors = log_frame(robot.position, robot.R.T)
        frame_origins += origins
        frame_vectors += vectors
        frame_colors.append([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

        positions.append(robot.position)

    rr.log("world/trajectory", rr.Points3D(positions), timeless=True)
    rr.log(
        "world/frames",
        rr.Arrows3D(origins=frame_origins, vectors=frame_vectors, colors=frame_colors),
        timeless=True,
    )
