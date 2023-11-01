from typing import Any

import numpy as np

from viosim.sensors import Camera


class Robot:
    def __init__(self, trajectory: Any, frequency: float, camera: Camera) -> None:
        self.trajectory = trajectory
        point, rotation = self.trajectory.pop()

        self.position = point
        self.velocity = np.array([0, 0, 0])
        self.acceleration = np.array([0, 0, 0])
        self.rotation = rotation
        self.angle_velocity = np.array([0, 0, 0])

        self.camera = camera
        self.camera.update_pose(self.position, self.rotation)

        self.frequency = frequency
        self.moving = True
        self.clock = 0

    @property
    def R_WtoR(self) -> np.array:
        return self.rotation.as_matrix()

    def init(self, world):
        self.camera.capture(world)

    def step(self, world):
        if len(self.trajectory) == 0:
            self.moving = False
            return

        point, rotation = self.trajectory.pop()

        self.angle_velocity = (
            self.rotation.as_euler("xyz") - rotation.as_euler("xyz")
        ) / self.frequency
        self.rotation = rotation

        new_position = point
        new_velocity = (self.position - new_position) / self.frequency
        new_acceleration = (self.velocity - new_velocity) / self.frequency

        self.position = new_position
        self.velocity = new_velocity
        self.acceleration = new_acceleration

        self.clock += self.frequency
        self.camera.update_pose(self.position, self.rotation)
        self.camera.capture(world)
