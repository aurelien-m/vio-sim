from __future__ import annotations

from dataclasses import dataclass
from random import randint, random

import numpy as np

import viosim.sensors


@dataclass
class WorldFeature:
    position: np.ndarray
    id: int


class World:
    def __init__(
        self,
        observer: viosim.sensors.Camera,
        min_observable_points: int = 10,
        min_depth: float = 2.0,
        max_depth: float = 20.0,
    ):
        self.observer = observer
        self.features = []

        self.min_obs_points = min_observable_points
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.last_id = 0

    def refresh(self):
        observations = len(self.observer.capture_data.observations)
        if observations < self.min_obs_points:
            for _ in range(self.min_obs_points - observations):
                x = randint(0, self.observer.width - 1)
                y = randint(0, self.observer.height - 1)
                depth = random() * (self.max_depth - self.min_depth) + self.min_depth

                position = self.observer.to_world(x, y, depth)
                self.features.append(WorldFeature(position, self.last_id))
                self.last_id += 1
