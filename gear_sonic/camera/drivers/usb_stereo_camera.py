"""Side-by-side USB stereo camera driver using OpenCV.

Many head-mounted stereo USB cameras expose a single UVC stream at 1280x480
(left eye | right eye).  This driver splits each frame into two RGB images
published as ``ego_view_left`` and ``ego_view_right``.
"""

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition


class USBStereoCameraConfig:
    """Configuration for a side-by-side USB stereo camera."""

    image_dim: tuple[int, int] = (1280, 480)
    fps: int = 30
    device_index: int = 0
    left_key: str = "ego_view_left"
    right_key: str = "ego_view_right"


class USBStereoCameraSensor(Sensor):
    """Sensor for side-by-side USB stereo cameras."""

    def __init__(
        self,
        config: USBStereoCameraConfig = USBStereoCameraConfig(),
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        device_index: int | None = None,
    ):
        self.config = config
        self.mount_position = mount_position
        self.left_key = config.left_key
        self.right_key = config.right_key

        idx = device_index if device_index is not None else config.device_index

        # Add V4L2 backend flag and MJPEG format for higher FPS on USB 2.0
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB stereo camera at index {idx}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.image_dim[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.image_dim[1])
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[{mount_position}] Warming up USB stereo camera...")
        for _ in range(10):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.1)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width < config.image_dim[0]:
            self.cap.release()
            raise RuntimeError(
                f"USB stereo camera at index {idx} opened at {width}x{height}, "
                f"expected at least {config.image_dim[0]}x{config.image_dim[1]}. "
                "Try a different /dev/video index or check v4l2 formats."
            )

        self._eye_width = width // 2
        self._eye_height = height

        print(f"[{mount_position}] USB stereo camera opened at index {idx}")
        print(f"  Resolution: {width}x{height} -> {self._eye_width}x{self._eye_height} per eye")
        print(f"  FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"  Stream keys: {self.left_key}, {self.right_key}")

    def read(self) -> dict[str, Any] | None:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print(f"[{self.mount_position}] USB stereo camera read failed: ret={ret}")
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mid = frame_rgb.shape[1] // 2
        left_rgb = frame_rgb[:, :mid]
        right_rgb = frame_rgb[:, mid:]
        timestamp = time.time()

        return {
            "timestamps": {self.left_key: timestamp, self.right_key: timestamp},
            "images": {self.left_key: left_rgb, self.right_key: right_rgb},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        from gear_sonic.camera.sensor_server import ImageMessageSchema

        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        return gym.spaces.Dict(
            {
                self.left_key: gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._eye_height, self._eye_width, 3),
                    dtype=np.uint8,
                ),
                self.right_key: gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._eye_height, self._eye_width, 3),
                    dtype=np.uint8,
                ),
            }
        )

    def close(self):
        if self.cap is not None:
            self.cap.release()
