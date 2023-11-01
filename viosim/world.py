from random import randint, random

from viosim.sensors import Camera


class World:
    def __init__(
        self,
        observer: Camera,
        min_observable_points: int = 10,
        min_depth: float = 2.0,
        max_depth: float = 20.0,
    ):
        self.observer = observer
        self.features = []

        self.min_obs_points = min_observable_points
        self.min_depth = min_depth
        self.max_depth = max_depth

    def refresh(self):
        if self.observer.observation_count < self.min_obs_points:
            for _ in range(self.min_obs_points - self.observer.observation_count):
                x = randint(0, self.observer.width - 1)
                y = randint(0, self.observer.height - 1)
                depth = random() * (self.max_depth - self.min_depth) + self.min_depth
                self.features.append(self.observer.to_world(x, y, depth))
