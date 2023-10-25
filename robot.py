from typing import List

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from utils import normalize


def rot_from_vect(vect: np.ndarray):
    theta = np.arccos(vect[2] / np.linalg.norm(vect))
    phi = np.arctan2(vect[1], vect[0])
    psi = 0.0
    R = Rotation.from_euler("xyz", [theta, phi, psi]).as_matrix()
    return R


def vector_to_rotation_matrix(v):
    v = normalize(v)

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

    points = [control_points[0]]
    orientations = []

    for i in range(1, len(t_new)):
        x, y, z = (
            cubic_spline_x(t_new[i]),
            cubic_spline_y(t_new[i]),
            cubic_spline_z(t_new[i]),
        )
        points.append(np.array([x, y, z]))

        R = rot_from_vect(points[i] - points[i - 1])
        orientations.append(R)

    return points[1:], orientations


class IMU:
    def __call__(
        self,
        gyro_noise_std: float = 0.1,
        accel_noise_std: float = 0.1,
        initial_gyro_bias: np.array = np.array([0, 0, 0]),
        initial_accel_bias: np.array = np.array([0, 0, 0]),
        gravity: float = -9.81,
    ) -> "IMU":
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gravity = np.array([[0], [0], [gravity]])
        self.gyroscope = np.array([0, 0, 0])
        self.accelerometer = np.array([0, 0, 0])
        self.gyro_bias = initial_gyro_bias
        self.accel_bias = initial_accel_bias

    def step(self, angle_velocity: np.array, acceleration: np.array, R: np.array):
        # TODO: Take the bias random walk into account

        n_g = np.random.normal(0, self.gyro_noise_std, 3)
        self.gyroscope = angle_velocity + self.gyro_bias + n_g

        n_a = np.random.normal(0, self.accel_noise_std, 3)
        self.accelerometer = R @ (acceleration - self.gravity) + self.accel_bias + n_a


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
        self.velocity = np.array([0, 0, 0])
        self.acceleration = np.array([0, 0, 0])
        self.clock = 0

        self.orientation = np.array([0, 0, 0])
        self.angle_velocity = np.array([0, 0, 0])

        self.frequency = frequency
        self.moving = True

    def step(self):
        if len(self.points) == 0:
            self.moving = False
            return

        self.R = self.orientations.pop(0)
        new_orientation = Rotation.from_matrix(self.R).as_euler("xyz")
        self.angle_velocity = (new_orientation - self.orientation) / self.frequency
        self.orientation = new_orientation

        new_position = self.points.pop(0)
        new_velocity = (new_position - self.position) / self.frequency
        new_acceleration = (new_velocity - self.velocity) / self.frequency

        self.position = new_position
        self.velocity = new_velocity
        self.acceleration = new_acceleration
        self.clock += self.frequency
