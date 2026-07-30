"""
Constant-velocity Kalman filter for pupil tracking (KTCal paper, Sec 3.1).

State vector: [x, y, vx, vy]  (position + velocity, normalized image coords in [0,1]).
Measurement : [x, y]         (pupil coordinate from NCC match or STNet).

Reset / rollback logic follows review reply W3:
  - When a reset is triggered (deviation sigma > tau3 against STNet observation),
    velocity is re-derived from the previous valid pupil position, or set to 0
    if no history exists; covariance is reset to diag([10, 10, 10, 10]).
"""
import numpy as np


class KalmanFilter2D:
    def __init__(self, dt=1.0, process_noise=1e-3, measurement_noise=1e-2,
                 velocity_decay=1.0):
        self.dt = dt
        self.q = process_noise
        self.r = measurement_noise
        self.velocity_decay = velocity_decay

        # State transition (constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, velocity_decay, 0],
            [0, 0, 0, velocity_decay],
        ], dtype=np.float64)

        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        self.Q = np.eye(4, dtype=np.float64) * self.q
        self.R = np.eye(2, dtype=np.float64) * self.r

        self.x = None          # state estimate (4,)
        self.P = None          # covariance (4,4)
        self.prev_pos = None   # last valid (x, y) for velocity re-derivation

    def is_initialized(self):
        return self.x is not None

    def initialize(self, pos):
        """pos: (x, y) normalized coordinate."""
        self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0])
        self.prev_pos = np.array(pos, dtype=np.float64)

    def predict(self):
        """Advance state; returns predicted (x, y)."""
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement, r_scale=1.0):
        """Correct state with measurement (x, y); returns corrected (x, y).

        r_scale scales the measurement noise R: a SMALL r_scale means the
        observation is trusted more (larger Kalman gain). Callers pass a small
        r_scale when the observation is high-confidence (e.g. high NCC score or
        the global STNet localization)."""
        if self.x is None:
            self.initialize(measurement)
            return self.x[:2].copy()
        z = np.array(measurement, dtype=np.float64)
        y = z - self.H @ self.x                         # innovation
        S = self.H @ self.P @ self.H.T + self.R * r_scale
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.prev_pos = self.x[:2].copy()
        return self.x[:2].copy()

    def reset(self, pos, previous_pos=None, velocity=None):
        """
        Threshold-triggered reset (review reply W3). By default velocity is
        derived from the latest valid position for backward compatibility.
        Callers may provide an explicit previous-frame position or velocity to
        evaluate alternative reset policies without changing the default.
        """
        pos = np.array(pos, dtype=np.float64)
        if velocity is not None:
            vx, vy = np.asarray(velocity, dtype=np.float64)
        else:
            velocity_anchor = self.prev_pos if previous_pos is None else np.asarray(
                previous_pos, dtype=np.float64)
            if velocity_anchor is not None:
                vx = (pos[0] - velocity_anchor[0]) / self.dt
                vy = (pos[1] - velocity_anchor[1]) / self.dt
            else:
                vx, vy = 0.0, 0.0
        self.x = np.array([pos[0], pos[1], vx, vy], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0])
        self.prev_pos = pos.copy()
        return self.x[:2].copy()


if __name__ == '__main__':
    kf = KalmanFilter2D()
    kf.initialize((0.5, 0.5))
    for step in range(5):
        pred = kf.predict()
        corr = kf.update((0.5 + 0.01 * step, 0.5 + 0.01 * step))
        print(f'step {step}: pred={pred}, corr={corr}')
