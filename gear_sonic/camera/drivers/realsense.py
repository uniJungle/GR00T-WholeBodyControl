"""Intel RealSense camera driver.

Requires the ``pyrealsense2`` SDK — install with::

    pip install pyrealsense2

See https://github.com/IntelRealSense/librealsense for hardware-specific instructions.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import pyrealsense2 as rs

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class RealSenseConfig:
    """Configuration for the RealSense camera."""

    depth_image_dim: tuple[int, int] = (640, 480)
    color_image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    # Match image_server_dex1: wrist D405 usually only needs RGB.
    enable_depth: bool = False
    rotate_180: bool = False
    mount_position: str = CameraMountPosition.EGO_VIEW.value


class RealSenseSensor(Sensor, SensorServer):
    """Sensor for Intel RealSense depth cameras."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: RealSenseConfig = RealSenseConfig(),
        id: int = 0,
        serial_number: str | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        # Best-effort listing only. Do NOT require the target serial to appear
        # here: under multi-thread staggered init (composed_camera),
        # query_devices() can temporarily omit a device that is still usable
        # via config.enable_device(serial), matching image_server_dex1 behavior.
        try:
            devices = list(rs.context().query_devices())
            devices_sorted = sorted(
                devices, key=lambda x: x.get_info(rs.camera_info.serial_number)
            )
            print(f"RealSense devices currently visible: {len(devices_sorted)}")
            for device in devices_sorted:
                print(f"Device: {device.get_info(rs.camera_info.name)}")
                print(
                    f"    Serial number: {device.get_info(rs.camera_info.serial_number)}"
                )
                print(
                    f"    Firmware version: "
                    f"{device.get_info(rs.camera_info.firmware_version)}"
                )
        except Exception as e:
            devices_sorted = []
            print(f"[WARN] RealSense query_devices failed (continuing): {e}")

        if serial_number is not None:
            selected_serial = str(serial_number)
        else:
            if not devices_sorted:
                raise RuntimeError("No RealSense devices found")
            if id < 0 or id >= len(devices_sorted):
                raise RuntimeError(
                    f"RealSense device index {id} out of range "
                    f"(found {len(devices_sorted)} device(s))"
                )
            selected_serial = devices_sorted[id].get_info(rs.camera_info.serial_number)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(selected_serial)

        # Prefer BGR8 to match OpenCV / image_server_dex1 convention.
        try:
            self.config.enable_stream(
                rs.stream.color,
                config.color_image_dim[0],
                config.color_image_dim[1],
                rs.format.bgr8,
                config.fps,
            )
            if config.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth,
                    config.depth_image_dim[0],
                    config.depth_image_dim[1],
                    rs.format.z16,
                    config.fps,
                )
            self.pipeline.start(self.config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to start RealSense pipeline for serial={selected_serial}: {e}"
            )

        self._realsense_config = config
        self._run_as_server = run_as_server
        self.mount_position = mount_position
        self.serial_number = selected_serial
        if self._run_as_server:
            self.start_server(port)
        print(f"Done initializing RealSense sensor: {selected_serial}")

    def read(self) -> dict[str, Any] | None:
        try:
            frames = self.pipeline.wait_for_frames()
        except Exception as e:
            print(f"ERROR! Failed to wait for frames: {e}")
            return None

        color_frame = frames.get_color_frame()
        if not color_frame:
            print("WARNING! No color frame")
            return None

        try:
            color_image = np.asanyarray(color_frame.get_data())
        except Exception as e:
            print(f"ERROR! Failed to convert color frame: {e}")
            return None

        if color_image.size == 0:
            print("WARNING! Empty color image")
            return None

        if self._realsense_config.rotate_180:
            color_image = cv2.rotate(color_image, cv2.ROTATE_180)

        current_time = time.time()
        timestamps = {self.mount_position: current_time}
        images = {self.mount_position: color_image}

        if self._realsense_config.enable_depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                print("WARNING! No depth frame")
                return None
            try:
                depth_image = np.asanyarray(depth_frame.get_data())
            except Exception as e:
                print(f"ERROR! Failed to convert depth frame: {e}")
                return None
            if depth_image.size == 0:
                print("WARNING! Empty depth image")
                return None
            if self._realsense_config.rotate_180:
                depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
            timestamps[f"{self.mount_position}_depth"] = current_time
            images[f"{self.mount_position}_depth"] = depth_image

        return {"timestamps": timestamps, "images": images}

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        spaces = {
            "color_image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    self._realsense_config.color_image_dim[1],
                    self._realsense_config.color_image_dim[0],
                    3,
                ),
                dtype=np.uint8,
            ),
        }
        if self._realsense_config.enable_depth:
            spaces["depth_image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    self._realsense_config.depth_image_dim[1],
                    self._realsense_config.depth_image_dim[0],
                    1,
                ),
                dtype=np.uint16,
            )
        return gym.spaces.Dict(spaces)

    def close(self):
        if self._run_as_server:
            self.stop_server()
        self.pipeline.stop()

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            read_result = self.read()
            if read_result is None:
                continue
            self.send_message({self.mount_position: self.serialize(read_result)})
