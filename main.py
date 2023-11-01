import numpy as np
import rerun as rr
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from viosim import Robot, World
from viosim.sensors import Camera
from viosim.trajectories import CubicSpineTrajectory


class RerunRobotLogger:
    def __init__(self):
        rr.init("VIO Simulator", spawn=True)
        self.robot_frame_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        self.positions = []

    def log(self, robot: Robot, world: World) -> None:
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
        rr.log("world/points", rr.Points3D(self.positions), timeless=True)

        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=robot.camera.position,
                mat3x3=robot.camera.rotation.as_matrix().T,
            ),
        )
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                focal_length=robot.camera.focal_px,
                width=robot.camera.width,
                height=robot.camera.height,
            ),
        )
        rr.log("world/camera/image", rr.Image(robot.camera.image))

        rr.log("world/features", rr.Points3D(world.features))


if __name__ == "__main__":
    logger = RerunRobotLogger()

    trajectory = CubicSpineTrajectory(
        control_points=[[0, 0, 0], [13, 8, 1], [22, 12, 6], [27, 24, 1]],
        target_spacing=0.5,
    )
    camera = Camera(
        pos_CinR=np.array([0, 0, 0]),
        rot_RtoC=Rotation.from_euler("xyz", [0, -np.pi / 2, -np.pi / 2]),
    )
    robot = Robot(trajectory, frequency=0.1, camera=camera)

    world = World(camera)
    world.refresh()

    robot.init(world)
    logger.log(robot, world)

    while robot.moving:
        world.refresh()  # Refresh the world first
        robot.step(world)  # Then step the robot accordingly

        logger.log(robot, world)
