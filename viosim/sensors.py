import numpy as np


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
