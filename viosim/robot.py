from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


class Robot:
    def __init__(
        self,
        trajectory: Any,
        frequency: float = 0.1,
    ) -> None:
        self.trajectory = trajectory

        self.position = np.array([0, 0, 0])
        self.velocity = np.array([0, 0, 0])
        self.acceleration = np.array([0, 0, 0])
        self.rotation = Rotation.from_euler("xyz", [0, 0, 0])
        self.angle_velocity = np.array([0, 0, 0])

        self.frequency = frequency
        self.moving = True
        self.clock = 0

    @property
    def R_WtoR(self) -> np.array:
        return self.rotation.as_matrix()

    def step(self):
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
