import numpy as np
import rerun as rr
from numpy.linalg import norm

from viosim import Robot
from viosim.trajectories import CubicSpineTrajectory


class RerunRobotLogger:
    def __init__(self):
        rr.init("VIO Simulator", spawn=True)
        self.robot_frame_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        self.positions = []

    def log(self, robot: Robot) -> None:
        rr.set_time_seconds("robot_clock", robot.clock)
        rr.log("robot/velocity", rr.Tensor(robot.velocity))
        rr.log("robot/acceleration", rr.Tensor(robot.acceleration))
        rr.log("robot/angle_velocity", rr.Tensor(robot.angle_velocity))

        rr.log("plot/velocity", rr.TimeSeriesScalar(norm(robot.velocity)))
        rr.log("plot/acceleration", rr.TimeSeriesScalar(norm(robot.acceleration)))

        x = robot.R_WtoR.T @ np.array([1, 0, 0])
        y = robot.R_WtoR.T @ np.array([0, 1, 0])
        z = robot.R_WtoR.T @ np.array([0, 0, 1])

        origin = [robot.position, robot.position, robot.position]
        vector = [x, y, z]
        rr.log(
            "world/frame",
            rr.Arrows3D(origins=origin, vectors=vector, colors=self.robot_frame_colors),
        )

        self.positions.append(robot.position)

    def log_trajectory(self) -> None:
        rr.log("world/points", rr.Points3D(self.positions), timeless=True)


if __name__ == "__main__":
    trajectory = CubicSpineTrajectory(
        control_points=[[0, 0, 0], [13, 8, 1], [22, 12, 6], [27, 24, 1]],
        target_spacing=0.5,
    )
    robot = Robot(trajectory)
    logger = RerunRobotLogger()

    while robot.moving:
        robot.step()
        logger.log(robot)

    logger.log_trajectory()
