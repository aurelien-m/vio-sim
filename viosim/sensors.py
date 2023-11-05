from typing import List

import numpy as np
from scipy.spatial.transform import Rotation


class Camera:
    def __init__(
        self,
        pos_CinR: np.ndarray,
        rot_RtoC: Rotation,
        principal_point: List[float] = (320, 320),
        width: float = 640,
        height: float = 640,
        focal: float = 0.002,
        pixel_size: List[float] = (2e-6, 2e-6),
    ) -> "Camera":
        self.pos_CinR = pos_CinR
        self.rot_RtoC = rot_RtoC

        self.width = width
        self.height = height
        self.focal_px = focal / pixel_size[0]
        self.K = np.array(
            [
                [focal / pixel_size[0], 0, principal_point[0]],
                [0, focal / pixel_size[1], principal_point[1]],
                [0, 0, 1],
            ]
        )

        self.observation_count = 0

    @property
    def T_inW(self):
        return self.position.reshape((3, 1))

    @property
    def R_WtoC(self):
        return self.rotation.as_matrix()

    def update_pose(self, pos_RinW: np.ndarray, rot_RtoW: Rotation) -> None:
        self.position = pos_RinW + rot_RtoW.as_matrix() @ self.pos_CinR
        self.rotation = self.rot_RtoC * rot_RtoW

    def capture(self, world: any) -> None:
        image = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        observation_count = 0

        for feature in world.features:
            xy = self.to_xy(feature)
            if xy is None:
                continue

            x, y = xy
            image[int(y), int(x)] = [255, 255, 255]
            observation_count += 1

        self.image = image
        self.observation_count = observation_count

    def to_xy(self, point_in_world: List[float]) -> np.ndarray:
        point_in_image = self.K @ self.to_cam(point_in_world).reshape((3, 1))
        x, y, depth = point_in_image.flatten()
        x, y = int(x / depth), int(y / depth)

        if not (0 <= x < self.width and 0 <= y < self.height) or depth < 0:
            return None
        return np.array([x, y])

    def to_cam(self, point_in_world: List[float]) -> np.ndarray:
        point_in_world = np.array(point_in_world).reshape((3, 1))
        return self.R_WtoC @ (point_in_world - self.T_inW)

    def to_world(self, x: int, y: int, depth: float) -> np.ndarray:
        point_in_image = np.array([x, y, 1]) * depth
        point_in_camera = np.linalg.inv(self.K) @ point_in_image
        return self.R_WtoC.T @ point_in_camera + self.position


class IMU:
    def __init__(
        self,
        gyro_noise_std: float = 0.01,
        accel_noise_std: float = 0.01,
        initial_gyro_bias: np.array = np.array([0.0, 0.0, 0.0]),
        gyro_bias_random_walk_std: float = 0.01,
        initial_accel_bias: np.array = np.array([0.0, 0.0, 0.0]),
        accel_bias_random_walk_std: float = 0.01,
        gravity: float = -9.81,
    ) -> "IMU":
        # TODO: these should be function of time somehow
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gyro_bias_std = gyro_bias_random_walk_std
        self.accel_bias_std = accel_bias_random_walk_std

        self.gravity = np.array([[0], [0], [gravity]])
        self.gyroscope = np.array([0, 0, 0])
        self.accelerometer = np.array([0, 0, 0])
        self.gyro_bias = initial_gyro_bias
        self.accel_bias = initial_accel_bias

    def init(self, position: np.ndarray, rotation: Rotation):
        self.position = position
        self.rotation = rotation
        self.velocity = np.array([0, 0, 0])
        self.acceleration = np.array([0, 0, 0])
        self.angle_velocity = np.array([0, 0, 0])

    def update(self, position: np.ndarray, rotation: Rotation, dt: float):
        self.gyro_bias += np.random.normal(self.gyro_bias, self.gyro_bias_std, 3)
        self.accel_bias += np.random.normal(self.accel_bias, self.accel_bias_std, 3)

        self.angle_velocity = (
            (self.rotation).as_euler("xyz") - rotation.as_euler("xyz")
        ) / dt
        self.rotation = rotation

        new_velocity = (position - self.position) / dt
        self.position = position

        self.acceleration = (self.velocity - new_velocity) / dt
        self.velocity = new_velocity

        n_g = np.random.normal(0, self.gyro_noise_std, 3)
        self.gyroscope = self.angle_velocity + self.gyro_bias + n_g
        n_a = np.random.normal(0, self.accel_noise_std, 3)
        self.accelerometer = (
            self.rotation.as_matrix() @ (self.acceleration - self.gravity)
            + self.accel_bias
            + n_a
        )
