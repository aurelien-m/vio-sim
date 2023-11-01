from typing import List
import numpy as np
from scipy.spatial.transform import Rotation


class Camera:
    def __init__(
        self,
        pos_CinR: np.ndarray,
        rot_RtoC: Rotation,
        principal_point: List[float] = (320, 320),
        image_size: List[float] = (640, 640),
        flocal: float = 0.002,
        pixel_size: List[float] = (2e-6, 2e-6),
    ) -> "Camera":
        self.pos_CinR = pos_CinR
        self.rot_RtoC = rot_RtoC
        self.image_size = image_size
        self.K = np.array(
            [
                [flocal / pixel_size[0], 0, principal_point[0]],
                [0, flocal / pixel_size[1], principal_point[1]],
                [0, 0, 1],
            ]
        )

    @property
    def R_(self):
        pass

    def update_pose(self, pos_RinW: np.ndarray, rot_RtoW: Rotation) -> None:
        self.position = pos_RinW + rot_RtoW.as_matrix() @ self.pos_CinR
        self.rotation = self.rot_RtoC * rot_RtoW

    def to_xy(self, point_in_world: List[float]):
        point_in_world = np.vstack((np.array(point_in_world).reshape((3, 1)), 1))
        projection_matrix = np.hstack(
            (self.rotation.as_matrix(), self.position.reshape((3, 1)))
        )

        point_in_image = self.K @ projection_matrix @ point_in_world
        point_in_image = (point_in_image / point_in_image[2])[:2].flatten()

        if (
            point_in_image[0] < 0
            or point_in_image[0] > self.image_size[0]
            or point_in_image[1] < 0
            or point_in_image[1] > self.image_size[1]
        ):
            return None
        return point_in_image


class IMU:
    def __init__(
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
