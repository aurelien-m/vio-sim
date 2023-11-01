from typing import List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


def rotation_from_direction(vect: np.ndarray) -> Rotation:
    theta = np.arccos(vect[2] / np.linalg.norm(vect))
    phi = np.arctan2(vect[1], vect[0])
    R_x = Rotation.from_euler("xyz", [np.pi / 2, 0.0, 0.0])
    return R_x * Rotation.from_euler("xyz", [theta, phi, 0.0])


class CubicSpineTrajectory:
    def __init__(self, control_points: List[List], target_spacing: float):
        distance = np.sum(np.linalg.norm(np.diff(control_points, axis=0), axis=1))
        num_samples = int(distance / target_spacing)

        t = range(len(control_points))
        x = [point[0] for point in control_points]
        y = [point[1] for point in control_points]
        z = [point[2] for point in control_points]

        cubic_spline_x = CubicSpline(t, x)
        cubic_spline_y = CubicSpline(t, y)
        cubic_spline_z = CubicSpline(t, z)

        t = np.linspace(0, len(control_points) - 1, num_samples)
        self.points = []
        self.rotations = []

        for i in range(len(t)):
            x = cubic_spline_x(t[i])
            y = cubic_spline_y(t[i])
            z = cubic_spline_z(t[i])

            self.points.append(np.array([x, y, z]))

        for i in range(len(self.points) - 1):
            self.rotations.append(
                rotation_from_direction(self.points[i + 1] - self.points[i])
            )
        self.rotations.append(self.rotations[-1])

    def __len__(self) -> int:
        return len(self.points)

    def pop(self) -> Tuple[np.ndarray, Rotation]:
        return self.points.pop(0), self.rotations.pop(0)
