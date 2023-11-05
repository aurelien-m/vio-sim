from typing import Any

import numpy as np

from viosim.sensors import Camera, IMU


class Robot:
    def __init__(self, trajectory: Any, frequency: float, camera: Camera) -> None:
        self.trajectory = trajectory
        position, rotation = self.trajectory.pop()

        self.frequency = frequency
        self.moving = True
        self.clock = 0

        self.camera = camera
        self.camera.update_pose(position, rotation)
        self.imu = IMU()
        self.imu.init(position, rotation)

    @property
    def R_WtoR(self) -> np.array:
        return self.imu.rotation.as_matrix()

    @property
    def position(self) -> np.array:
        return self.imu.position

    @property
    def velocity(self) -> np.array:
        return self.imu.velocity

    @property
    def acceleration(self) -> np.array:
        return self.imu.acceleration

    @property
    def angle_velocity(self) -> np.array:
        return self.imu.angle_velocity

    @property
    def rotation(self) -> np.array:
        return self.camera.rotation.as_matrix()

    def init(self, world):
        self.camera.capture(world)

    def step(self, world):
        if len(self.trajectory) == 0:
            self.moving = False
            return

        position, rotation = self.trajectory.pop()
        self.clock += self.frequency

        self.camera.update_pose(position, rotation)
        self.camera.capture(world)

        self.imu.update(position, rotation, self.frequency)
