from typing import List

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def find_R_between_vectors(v1, v2):
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    skew = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    R = np.identity(3) + skew + np.dot(skew, skew) * (1 - c) / (s**2)
    return R


def vector_to_rotation_matrix(v):
    v = normalize_vector(v)

    x, y, z = v
    c = 1 / (1 + x)

    R = np.array(
        [
            [1 - x * x * c, -x * y * c, -x * z * c],
            [-x * y * c, 1 - y * y * c, -y * z * c],
            [-x * z * c, -y * z * c, 1 - z * z * c],
        ]
    )

    return R


def generate_trajectory(control_points: List[List], target_spacing: float):
    distance = np.sum(np.linalg.norm(np.diff(control_points, axis=0), axis=1))
    num_samples = int(distance / target_spacing)

    x = [point[0] for point in control_points]
    y = [point[1] for point in control_points]
    z = [point[2] for point in control_points]

    t = range(len(control_points))
    cubic_spline_x = CubicSpline(t, x)
    cubic_spline_y = CubicSpline(t, y)
    cubic_spline_z = CubicSpline(t, z)

    t_new = np.linspace(0, len(control_points) - 1, num_samples)

    points = [np.array([0, 0, 0])]
    orientations = []

    for i in range(1, len(t_new)):
        x, y, z = (
            cubic_spline_x(t_new[i]),
            cubic_spline_y(t_new[i]),
            cubic_spline_z(t_new[i]),
        )
        points.append(np.array([x, y, z]))

        R = find_R_between_vectors(points[i] - points[i - 1], np.array([1, 0, 0]))
        orientations.append(R)

    return points[1:], orientations


class Robot:
    def __init__(
        self,
        trajectory: List[List],
        frequency: float = 0.1,
        average_velocity: float = 20,
    ) -> None:
        assert len(trajectory) > 0, "Trajectory must have at least one point"

        spacing = average_velocity * frequency
        self.points, self.orientations = generate_trajectory(trajectory, spacing)

        self.position = np.array([0, 0, 0])
        self.acceleration = np.array([0, 0, 0])
        self.velocity = np.array([0, 0, 0])
        self.clock = 0

        self.frequency = frequency
        self.moving = True

    def step(self):
        if len(self.points) == 0:
            self.moving = False
            return

        self.R = self.orientations.pop(0)

        new_position = self.points.pop(0)
        new_velocity = (new_position - self.position) / self.frequency
        new_acceleration = (new_velocity - self.velocity) / self.frequency

        self.position = new_position
        self.velocity = new_velocity
        self.acceleration = new_acceleration
        self.clock += self.frequency
