"""
Sonic VLA data exporter for G1 -- NO ROS 2 DEPENDENCY.

All data sources use ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (port 5557, from C++ zmq_output_handler)
  2. SMPL pose    -> ZMQ SUB on ``pose`` topic     (port 5556, from pico_manager_thread_server)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor

Robot config (``script_config`` in info.json) is read from the ``robot_config``
ZMQ topic re-published every ~2 s by the C++ process.  If the config is not
received within the timeout the exporter exits with an error.

Virtual environment setup (run from repo root):
    bash install_scripts/install_data_collection.sh
    source .venv_data_collection/bin/activate

Usage (from repo root):
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cup"
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "walk forward" --task-name pour
    # Optional session subdir: --dataset-name my_session
    #   -> <root>/<task_name>/<dataset_name>/
    # Default (no --dataset-name):
    #   -> <root>/<task_name>/
"""

from collections import deque
from dataclasses import dataclass
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro
import zmq
import sys
import os

# Add unitree_sdk2_python to path for AudioClient
# gear_sonic/scripts/ -> ../../external_dependencies/unitree_sdk2_python
_UNITREE_SDK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../external_dependencies/unitree_sdk2_python")
)
if _UNITREE_SDK_PATH not in sys.path:
    sys.path.insert(0, _UNITREE_SDK_PATH)

HAS_UNITREE_AUDIO = False
_UNITREE_AUDIO_IMPORT_ERROR = None
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    HAS_UNITREE_AUDIO = True
except Exception as e:
    _UNITREE_AUDIO_IMPORT_ERROR = e
    ChannelFactoryInitialize = None  # type: ignore
    AudioClient = None  # type: ignore

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
    get_stereo_ego_camera_features,
    get_stereo_ego_camera_modality_config,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity, quat_to_rot6d
from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    ZMQStateSubscriber,
    poll_robot_config_zmq,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SonicDataExporterConfig:
    """CLI config for the ROS-free Sonic data exporter."""

    # Dataset
    dataset_name: str | None = None
    """Optional subdirectory under task_name. If None, save directly to
    ``<root_output_dir>/<task_name>/`` (no date folder)."""

    task_name: str = "default_task"
    """Task specific directory name."""

    task_prompt: str = "demo"
    """Language task prompt."""

    root_output_dir: str = "outputs"
    """Root output directory."""

    data_collection_frequency: int = 50
    """Data collection frequency (Hz)."""


    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Sonic / SMPL pose (from pico_manager_thread_server)
    sonic_zmq_host: str = "localhost"
    """ZMQ host for Sonic SMPL pose messages."""

    sonic_zmq_port: int = 5556
    """ZMQ port for Sonic SMPL pose messages."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # Robot config
    robot_config_timeout: float = 0
    """Seconds to wait for the ZMQ robot_config message at startup (0 = wait forever)."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist). Requires cameras to be available."""

    record_stereo_ego: bool = False
    """Record head stereo as observation.images.ego_view_left / ego_view_right."""

    text_to_speech: bool = True
    """Use local pyttsx3 text-to-speech voice feedback (PC speaker)."""

    robot_tts: bool = True
    """Use G1 onboard AudioClient TTS over DDS (requires cyclonedds + unitree_sdk2py)."""

    dds_interface: str = "enp4s0"
    """Network interface connected to the robot (for Unitree DDS / AudioClient)."""

    eef: str = "dex3"
    """End-effector driver: 'dex3' (7-dim), 'brainco' (2-dim), or 'dex1' (1-dim)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_robot_audio_client(dds_interface: str, init_channel_factory: bool = True):
    """Create a G1 AudioClient with voice-mode API registration.

    Matches the working pattern from wbc_pico_record (VoiceModeClient + mode 3).

    If ``init_channel_factory`` is False, assume ChannelFactory was already
    initialized (e.g. by Brainco passive monitor on the same process).
    """
    if not HAS_UNITREE_AUDIO:
        raise RuntimeError(
            f"unitree_sdk2py unavailable: {_UNITREE_AUDIO_IMPORT_ERROR}. "
            "Install cyclonedds into .venv_data_collection "
            "(e.g. copy from .venv_teleop or: uv pip install cyclonedds==0.10.2)."
        )

    class VoiceModeClient(AudioClient):
        def Init(self):
            super().Init()
            # API 1008 is required for G1 voice mode switching.
            self._RegistApi(1008, 0)

        def SetVoiceMode(self, mode: int):
            import json as _json

            code, data = self._Call(1008, _json.dumps({"mode": mode}))
            return code, data

        def TtsMaker(self, text: str, speaker_id: int):
            # Upstream unitree_sdk2py does ``tts_index += tts_index`` which stays
            # at 0 forever when starting from 0; G1 then ignores repeated TTS.
            import json as _json
            from unitree_sdk2py.g1.audio.g1_audio_api import (
                ROBOT_API_ID_AUDIO_STOP_PLAY,
                ROBOT_API_ID_AUDIO_TTS,
            )

            # Stop any in-flight playback so the new phrase is not dropped.
            try:
                self._Call(ROBOT_API_ID_AUDIO_STOP_PLAY, _json.dumps({}))
            except Exception:
                pass
            self.SetVolume(100)
            self.tts_index = int(getattr(self, "tts_index", 0)) + 1
            parameter = _json.dumps(
                {"index": self.tts_index, "text": text, "speaker_id": int(speaker_id)}
            )
            code, _data = self._Call(ROBOT_API_ID_AUDIO_TTS, parameter)
            return code

    if init_channel_factory:
        ChannelFactoryInitialize(0, dds_interface)
    client = VoiceModeClient()
    client.SetTimeout(5.0)
    client.Init()
    client.SetVolume(100)
    code, _ = client.SetVoiceMode(3)
    print(f"[Audio] Voice mode set to 3 (code={code})")
    return client


class TimeDeltaException(Exception):
    def __init__(self, failure_count: int, reset_timeout_sec: float):
        self.failure_count = failure_count
        self.reset_timeout_sec = reset_timeout_sec
        self.message = f"{self.failure_count} failures in {self.reset_timeout_sec} seconds"
        super().__init__(self.message)


def unpack_pose_message(packed_data: bytes, topic: str = "pose") -> dict:
    """Unpack a single-frame packed message from pico_manager_thread_server.

    Wire format: [topic_prefix][1280-byte JSON header][concatenated binary fields]
    """
    HEADER_SIZE = 1280

    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic '{topic}'")

    offset = len(topic_bytes)
    if len(packed_data) < offset + HEADER_SIZE:
        raise ValueError(f"Packed data too small: {len(packed_data)} < {offset + HEADER_SIZE}")

    header_bytes = packed_data[offset : offset + HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx > 0:
        header_bytes = header_bytes[:null_idx]

    header = json.loads(header_bytes.decode("utf-8"))
    fields = header.get("fields", [])

    result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
    current_offset = offset + HEADER_SIZE
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": bool,
    }

    for field in fields:
        dtype = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        result[field["name"]] = (
            np.frombuffer(packed_data[current_offset : current_offset + n_bytes], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        current_offset += n_bytes

    return result


class TimingThresholdMonitor:
    def __init__(self, max_failures=3, reset_timeout_sec=5, time_delta=0.2, raise_exception=False):
        self.max_failures = max_failures
        self.reset_timeout_sec = reset_timeout_sec
        self.failure_count = 0
        self.last_failure_time = 0
        self.time_delta = time_delta
        self.raise_exception = raise_exception

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = 0

    def log_time_delta(self, time_delta_sec: float):
        time_delta = abs(time_delta_sec)
        if time_delta > self.time_delta:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

            if self.is_threshold_exceeded():
                print(
                    f"Time delta exception: {self.failure_count} failures in "
                    f"{self.reset_timeout_sec} seconds, time delta: {time_delta}"
                )
                if self.raise_exception:
                    raise TimeDeltaException(self.failure_count, self.reset_timeout_sec)
        else:
            # Clear failures if we get a good reading
            self.reset()

    def is_threshold_exceeded(self):
        if time.monotonic() - self.last_failure_time > self.reset_timeout_sec:
            self.reset()
            return False
        if self.failure_count >= self.max_failures:
            return True
        return False


# ---------------------------------------------------------------------------
# Data Collector
# ---------------------------------------------------------------------------


class GrootDataCollector:
    """Collects data from G1 robot in Sonic CPP + SMPL mode -- no ROS 2.

    Data sources (all ZMQ):
      - ``g1_debug`` topic        -> proprio (body_q, hand_q, actions, base_quat, ...)
      - ``pose`` topic            -> SMPL pose (smpl_joints, body_quat_w, hand_joints, ...)
      - ``planner`` topic         -> planner commands (vr_position, vr_orientation, ...)
      - ``manager_state`` topic   -> current stream mode + toggle flags
      - Camera client             -> ego-view images
    """

    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        data_exporter: Gr00tDataExporter,
        robot_model,
        text_to_speech=None,
        frequency: int = 20,
        sonic_data_zmq_host: str = "localhost",
        sonic_data_zmq_port: int = 5556,
        state_zmq_host: str = "localhost",
        state_zmq_port: int = 5557,
        robot_tts: bool = True,
        dds_interface: str = "enp4s0",
        eef: str = "dex3",
    ):
        self.text_to_speech = text_to_speech
        self.frequency = frequency
        self.loop_period = 1.0 / frequency
        self.data_exporter = data_exporter
        self.robot_model = robot_model
        self.eef = eef
        if eef == "brainco":
            self.hand_dim = 2
        elif eef == "dex1":
            self.hand_dim = 1
        else:
            self.hand_dim = 7

        # Unitree DDS ChannelFactory can only be initialized once per process.
        # Brainco passive monitor and G1 AudioClient both need it — init once here.
        dds_factory_ready = False
        need_dds = (self.eef == "brainco") or robot_tts
        if need_dds and ChannelFactoryInitialize is not None:
            try:
                ChannelFactoryInitialize(0, dds_interface)
                dds_factory_ready = True
                print(f"[DDS] ChannelFactory initialized on '{dds_interface}'.")
            except Exception as e:
                print(f"[DDS] ChannelFactoryInitialize failed: {e}")

        self.brainco_hand = None
        if self.eef == "brainco":
            from eef.brainco.brainco import Brainco

            self.brainco_hand = Brainco(
                passive=True,
                network_interface=dds_interface,
                init_channel_factory=not dds_factory_ready,
            )
            if not dds_factory_ready:
                dds_factory_ready = True
            print("[Brainco] Passive mode initialized for data recording (2-dim).")

        self._episode_state = EpisodeState()
        self._keyboard_listener = ZMQKeyboardSubscriber()

        self._image_subscriber = ComposedCameraClientSensor(server_ip=camera_host, port=camera_port)

        self.obs_act_buffer = deque(maxlen=100)
        self.latest_image_msg = None
        self.latest_proprio_msg = None
        self.latest_sonic_msg = None
        self.latest_planner_msg = None

        self.current_stream_mode = 0
        self._last_announced_stream_mode: int | None = None
        self._last_announced_locomotion_mode: int | None = None

        self._manager_toggle_dc = False
        self._manager_toggle_da = False

        self._state_subscriber = ZMQStateSubscriber(
            host=state_zmq_host,
            port=state_zmq_port,
        )

        self._sonic_zmq_ctx = None
        self._sonic_zmq_socket = None
        self._manager_zmq_socket = None
        try:
            self._sonic_zmq_ctx = zmq.Context()
            endpoint = f"tcp://{sonic_data_zmq_host}:{sonic_data_zmq_port}"

            # pose/planner can be high-rate; keep a modest queue for latest motion.
            self._sonic_zmq_socket = self._sonic_zmq_ctx.socket(zmq.SUB)
            self._sonic_zmq_socket.connect(endpoint)
            self._sonic_zmq_socket.setsockopt(zmq.RCVTIMEO, 100)
            self._sonic_zmq_socket.setsockopt(zmq.CONFLATE, 0)
            self._sonic_zmq_socket.setsockopt(zmq.RCVHWM, 200)
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "planner")

            # manager_state must NOT share the pose queue: after POSE_PAUSE→POSE,
            # pose flood used to drop mode=1 updates and leave exporter stuck at
            # mode=4. Dedicated socket keeps control messages reliable.
            # Do NOT use CONFLATE here: toggle_data_* are one-frame rising edges;
            # CONFLATE would overwrite True with later False and lose grip+A/B.
            self._manager_zmq_socket = self._sonic_zmq_ctx.socket(zmq.SUB)
            self._manager_zmq_socket.connect(endpoint)
            self._manager_zmq_socket.setsockopt(zmq.RCVTIMEO, 100)
            self._manager_zmq_socket.setsockopt(zmq.CONFLATE, 0)
            self._manager_zmq_socket.setsockopt(zmq.RCVHWM, 200)
            self._manager_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "manager_state")

            time.sleep(0.5)
            print(f"[Sonic] Connected to ZMQ at {endpoint}")
            print("[Sonic] Subscribed to: pose, planner (data) + manager_state (control)")
        except Exception as e:
            print(f"[Sonic] Warning: Failed to initialize ZMQ subscriber: {e}")
            self._sonic_zmq_socket = None
            self._manager_zmq_socket = None

        self.telemetry = Telemetry(window_size=100)
        self.sonic_timing_monitor = TimingThresholdMonitor(
            max_failures=3, reset_timeout_sec=5, time_delta=0.1
        )

        self._last_latency_log_time = 0.0
        self._initial_yaw = None

        self.audio_client = None
        if robot_tts:
            if not HAS_UNITREE_AUDIO:
                print(
                    f"[Audio] Disabled: cannot import unitree_sdk2py "
                    f"({_UNITREE_AUDIO_IMPORT_ERROR}). "
                    f"Looked in: {_UNITREE_SDK_PATH}"
                )
            else:
                try:
                    self.audio_client = _create_robot_audio_client(
                        dds_interface,
                        init_channel_factory=not dds_factory_ready,
                    )
                    print(
                        f"[Audio] Unitree AudioClient initialized on '{dds_interface}'."
                    )
                except Exception as e:
                    print(f"[Audio] Failed to initialize Unitree AudioClient: {e}")
                    self.audio_client = None

        print(f"Recording to {self.data_exporter.meta.root}")

    @property
    def current_episode_index(self):
        return self.data_exporter.episode_buffer["episode_index"]

    def _robot_say(self, text: str, speaker_id: int = 0) -> None:
        """Speak on the G1 speaker via AudioClient. Failures are logged, not raised."""
        if self.audio_client is None:
            print(f"[Audio] skip TtsMaker (no client): {text}")
            return
        try:
            code = self.audio_client.TtsMaker(text, speaker_id)
            print(f"[Audio] TtsMaker('{text}') code={code}")
        except Exception as e:
            print(f"[Audio] TtsMaker('{text}') failed: {e}")

    # Matches pico_manager StreamMode. Log-only here; pico speaks on the robot.
    # 3 (PLANNER_FROZEN_UPPER_BODY) / 5 (PLANNER_VR_3PT) intentionally omitted.
    _STREAM_MODE_TTS = {
        0: "模式关闭",           # OFF
        1: "进入遥操模式",       # POSE (A+X)
        2: "进入规划模式",       # PLANNER (A+X / pico disconnect)
        4: "遥操暂停",           # POSE_PAUSE (tap right Y)
    }

    def _announce_stream_mode(self, mode: int) -> None:
        """Track stream-mode changes; TTS is handled by pico_manager (reliable AudioClient)."""
        if self._last_announced_stream_mode is None:
            # First packet: sync silently, avoid speaking on exporter startup.
            self._last_announced_stream_mode = mode
            return
        if mode == self._last_announced_stream_mode:
            return
        prev = self._last_announced_stream_mode
        self._last_announced_stream_mode = mode
        if mode == 1 and prev == 4:
            text = "遥操返回"
        else:
            text = self._STREAM_MODE_TTS.get(mode)
        if text is None:
            print(f"[Mode] stream_mode {prev} -> {mode}: (no TTS, pico owns voice)")
            return
        # Log only — pico announces on the working speaker path.
        print(f"[Mode] stream_mode {prev} -> {mode}: {text} (TTS via pico)")

    def _announce_locomotion_mode(self, mode: int) -> None:
        """Track locomotion changes; SLOW_WALK TTS is spoken by pico_manager."""
        if self._last_announced_locomotion_mode is None:
            self._last_announced_locomotion_mode = mode
            return
        if mode == self._last_announced_locomotion_mode:
            return
        prev = self._last_announced_locomotion_mode
        self._last_announced_locomotion_mode = mode
        print(f"[Mode] locomotion_mode {prev} -> {mode} (TTS via pico if SLOW_WALK)")

    def _print_and_say(self, message: str, say: bool = True, blocking: bool = False):
        if self.text_to_speech is not None:
            self.text_to_speech.print_and_say(message, say, blocking=blocking)
        else:
            print(message)

    def _poll_state_zmq(self):
        """Poll the ``g1_debug`` ZMQ topic for robot state (non-blocking)."""
        msg = self._state_subscriber.get_msg(clear=True)
        if msg is None:
            return

        if msg.get("ros_timestamp", 0.0) == 0.0:
            msg["ros_timestamp"] = time.time()

        self.latest_proprio_msg = msg

    def _check_recording_commands(self):
        """Check keyboard + ZMQ toggle flags for recording commands.

        Start/stop/discard only apply in teleop (StreamMode.POSE). Planner-mode
        grip+A/B must not save or discard episodes.
        """
        key = self._keyboard_listener.read_msg()

        if self._manager_toggle_da:
            key = "x"
            self._manager_toggle_da = False
        elif self._manager_toggle_dc:
            key = "c"
            self._manager_toggle_dc = False

        # Keyboard c/x is also teleop-gated so planner sessions cannot archive data.
        in_teleop = int(getattr(self, "current_stream_mode", 0)) == 1
        if key in ("c", "x") and not in_teleop:
            self._print_and_say(
                "Record/discard ignored: not in teleop (POSE) mode",
                say=False,
            )
            return

        if key == "c":
            self._episode_state.change_state()
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self._initial_yaw = None
                if self.latest_image_msg is None:
                    self._print_and_say(
                        f"Started recording {self.current_episode_index} "
                        "(no camera yet — images will be blank until image server is up)",
                        blocking=False,
                    )
                else:
                    self._print_and_say(
                        f"Started recording {self.current_episode_index}", blocking=False
                    )
                self._robot_say("开始录制")
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                self._print_and_say("Stopping recording, preparing to save", blocking=False)
                # TTS deferred to _finalize_frame after save succeeds ("保存第i条数据").
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                self._print_and_say("Saved episode and back to idle state", blocking=False)
        elif key == "x":
            state = self._episode_state.get_state()
            if state in (self._episode_state.RECORDING, self._episode_state.NEED_TO_SAVE):
                discarded_idx = int(self.data_exporter.episode_buffer["episode_index"])
                buffer_size = self.data_exporter.episode_buffer.get("size", 0)
                if buffer_size > 0:
                    self.data_exporter.save_episode_as_discarded()
                else:
                    self._print_and_say(
                        "Discarded empty episode (no frames collected yet)",
                        say=False,
                    )
                self._episode_state.reset_state()
                self._initial_yaw = None
                self._print_and_say("Discarded episode", blocking=False)
                self._robot_say(f"丢弃第{discarded_idx + 1}条")
            else:
                self._print_and_say("Discard ignored: not recording", say=False)

    def _poll_sonic_zmq_messages(self):
        """Poll ZMQ for manager_state, pose, and planner (non-blocking).

        manager_state is on a dedicated socket so pause/resume mode updates are
        never starved by high-rate pose frames. Drain the full control queue each
        tick so one-frame toggle edges (grip+A/B) are not missed.
        """
        # 1) Control path first: drain all pending manager_state messages.
        if self._manager_zmq_socket is not None:
            for _ in range(64):
                try:
                    raw = self._manager_zmq_socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if raw.startswith(b"manager_state"):
                    self._handle_manager_state(raw)

        # 2) Motion path: drain a burst of pose/planner.
        if self._sonic_zmq_socket is None:
            return

        max_polls = 40
        for _ in range(max_polls):
            try:
                raw = self._sonic_zmq_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            if raw.startswith(b"planner"):
                self._handle_planner_message(raw)
            elif raw.startswith(b"pose"):
                self._handle_pose_message(raw)

    def _handle_manager_state(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="manager_state")
        except Exception:
            return

        if "stream_mode" in data:
            new_mode = int(data["stream_mode"].flat[0])
            self.current_stream_mode = new_mode
            # A+X (and other manager combos) change stream_mode in pico_manager;
            # announce here so we reuse the existing AudioClient (avoid dual DDS clients).
            self._announce_stream_mode(new_mode)

        if "locomotion_mode" in data:
            loco_mode = int(np.asarray(data["locomotion_mode"]).flat[0])
            self._announce_locomotion_mode(loco_mode)

        # Recording controls are teleop-only (StreamMode.POSE == 1). Ignore grip+A/B
        # edges published while pico is in planner / pause / VR3PT / off.
        in_teleop = int(getattr(self, "current_stream_mode", 0)) == 1
        if in_teleop and self._extract_bool(data, "toggle_data_collection"):
            self._manager_toggle_dc = True
        if in_teleop and self._extract_bool(data, "toggle_data_abort"):
            self._manager_toggle_da = True

    def _handle_planner_message(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="planner")
        except Exception:
            return

        planner_mode = int(data["mode"].flat[0]) if "mode" in data else 0
        planner_movement = (
            data["movement"].flatten().astype(np.float32)
            if "movement" in data and data["movement"].size == 3
            else np.zeros(3, dtype=np.float32)
        )
        planner_facing = (
            data["facing"].flatten().astype(np.float32)
            if "facing" in data and data["facing"].size == 3
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        planner_speed = float(data["speed"].flat[0]) if "speed" in data else -1.0
        planner_height = float(data["height"].flat[0]) if "height" in data else -1.0

        vr_3pt_position = None
        if "vr_position" in data and data["vr_position"].size == 9:
            vr_3pt_position = data["vr_position"].flatten().astype(np.float32)
        vr_3pt_orientation = None
        if "vr_orientation" in data and data["vr_orientation"].size == 12:
            vr_3pt_orientation = data["vr_orientation"].flatten().astype(np.float32)

        self.latest_planner_msg = {
            "planner_mode": planner_mode,
            "planner_movement": planner_movement,
            "planner_facing": planner_facing,
            "planner_speed": planner_speed,
            "planner_height": planner_height,
            "vr_3pt_position": vr_3pt_position,
            "vr_3pt_orientation": vr_3pt_orientation,
            "left_hand_joints": self._extract_hand_joints(data, "left_hand_joints"),
            "right_hand_joints": self._extract_hand_joints(data, "right_hand_joints"),
            "receive_timestamp": time.time(),
        }

    def _handle_pose_message(self, raw: bytes) -> None:
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27
        G1_R_WRIST_ROLL_IDX = 24
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28

        try:
            pose_data = unpack_pose_message(raw, topic="pose")
        except Exception as e:
            print(f"[Sonic] Error unpacking pose message: {e}")
            return

        try:
            if "smpl_joints" not in pose_data or len(pose_data["smpl_joints"].shape) != 3:
                return

            left_wrist_joints = None
            right_wrist_joints = None
            if "joint_pos" in pose_data and len(pose_data["joint_pos"].shape) == 2:
                joint_pos = pose_data["joint_pos"][0]
                left_wrist_joints = np.array(
                    [
                        joint_pos[G1_L_WRIST_ROLL_IDX],
                        joint_pos[G1_L_WRIST_PITCH_IDX],
                        joint_pos[G1_L_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )
                right_wrist_joints = np.array(
                    [
                        joint_pos[G1_R_WRIST_ROLL_IDX],
                        joint_pos[G1_R_WRIST_PITCH_IDX],
                        joint_pos[G1_R_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )

            frame_index = None
            if "frame_index" in pose_data:
                frame_index = np.array([pose_data["frame_index"].flat[0]], dtype=np.int64)

            smpl_pose = np.zeros(63, dtype=np.float32)
            if "smpl_pose" in pose_data:
                raw_pose = pose_data["smpl_pose"]
                if raw_pose.ndim == 3:
                    smpl_pose = raw_pose[0].flatten().astype(np.float32)
                elif raw_pose.ndim == 2:
                    smpl_pose = raw_pose.flatten().astype(np.float32)
                elif raw_pose.ndim == 1 and raw_pose.size == 63:
                    smpl_pose = raw_pose.astype(np.float32)

            left_hand_joints = self._extract_hand_joints(pose_data, "left_hand_joints")
            right_hand_joints = self._extract_hand_joints(pose_data, "right_hand_joints")

            vr_3pt_position = None
            if "vr_position" in pose_data and pose_data["vr_position"].size == 9:
                vr_3pt_position = pose_data["vr_position"].flatten().astype(np.float32)
            vr_3pt_orientation = None
            if "vr_orientation" in pose_data and pose_data["vr_orientation"].size == 12:
                vr_3pt_orientation = pose_data["vr_orientation"].flatten().astype(np.float32)

            self.latest_sonic_msg = {
                "smpl_joints": pose_data["smpl_joints"][0],
                "smpl_pose": smpl_pose,
                "body_quat_w": (
                    pose_data["body_quat_w"][0] if "body_quat_w" in pose_data else None
                ),
                "left_hand_joints": left_hand_joints,
                "right_hand_joints": right_hand_joints,
                "left_wrist_joints": left_wrist_joints,
                "right_wrist_joints": right_wrist_joints,
                "vr_3pt_position": vr_3pt_position,
                "vr_3pt_orientation": vr_3pt_orientation,
                "frame_index": frame_index,
                "receive_timestamp": time.time(),
            }
            # pose topic is only published in POSE mode. If exporter was stuck at
            # POSE_PAUSE (4) due to a dropped manager_state, clear it here.
            if self.current_stream_mode == 4:
                self.current_stream_mode = 1
                self._announce_stream_mode(1)
        except Exception as e:
            if not hasattr(self, "_sonic_error_count"):
                self._sonic_error_count = 0
            self._sonic_error_count += 1
            if self._sonic_error_count == 1 or self._sonic_error_count % 100 == 0:
                print(f"[Sonic] Error processing pose message: {e}")

    def _extract_hand_joints(self, pose_data: dict, key: str) -> np.ndarray:
        arr = pose_data.get(key)
        if arr is not None:
            if arr.ndim > 1:
                arr = arr[0]
            return arr.astype(np.float32)
        return np.zeros(self.hand_dim, dtype=np.float32)

    @staticmethod
    def _extract_bool(pose_data: dict, key: str) -> bool:
        val = pose_data.get(key)
        if val is None:
            return False
        if isinstance(val, np.ndarray):
            return bool(val.flat[0])
        return bool(val)

    def _log_latency_periodic(
        self,
        sonic_latency_ms: float | None = None,
    ):
        current_time = time.time()
        if current_time - self._last_latency_log_time >= 1.0:
            self._last_latency_log_time = current_time
            parts = []
            if sonic_latency_ms is not None:
                parts.append(f"Sonic Pose: {sonic_latency_ms:.1f}ms")
            if parts:
                print(f"[Latency] {', '.join(parts)}")

    def _blank_image_for_feature(self, feature_info: dict) -> np.ndarray:
        """Black RGB frame matching feature shape (used when camera is down)."""
        shape = feature_info.get("shape", [480, 640, 3])
        h, w = int(shape[0]), int(shape[1])
        c = int(shape[2]) if len(shape) > 2 else 3
        return np.zeros((h, w, c), dtype=np.uint8)

    def _add_images_to_frame_data(self, frame_data: dict) -> None:
        images = {}
        if self.latest_image_msg is not None:
            images = self.latest_image_msg.get("images") or {}

        missing = []
        for feature_name, feature_info in self.data_exporter.features.items():
            if feature_info.get("dtype") not in ["image", "video"]:
                continue
            image_key = feature_name.split(".")[-1]
            if image_key in images and images[image_key] is not None:
                frame_data[feature_name] = images[image_key]
            else:
                frame_data[feature_name] = self._blank_image_for_feature(feature_info)
                missing.append(image_key)

        if missing:
            now = time.monotonic()
            last = getattr(self, "_last_blank_image_log", 0.0)
            if now - last >= 2.0:
                self._last_blank_image_log = now
                self._print_and_say(
                    f"[Camera] missing {missing} — writing blank frames "
                    f"(other modalities still recorded)",
                    say=False,
                )

    def _finalize_frame(self, t_start: float) -> bool:
        t_end = time.monotonic()
        if t_end - t_start > (1 / self.frequency):
            print(f"DataExporter Missed: {t_end - t_start} sec")

        if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                # episode_index is scoped to the current dataset path (continues on resume).
                saved_episode_index = int(self.data_exporter.episode_buffer["episode_index"])
                self.data_exporter.save_episode()
                self.sonic_timing_monitor.reset()
                self._initial_yaw = None
                # 1-based for human speech; matches episode_XXXXXX under this save root.
                episode_no = saved_episode_index + 1
                self._print_and_say(f"Finished saving episode {saved_episode_index}")
                self._robot_say(f"保存第{episode_no}条数据")
            else:
                self._print_and_say("Skipping save: no frames collected", say=False)
            self._episode_state.change_state()
        return True

    def _add_data_frame(self):
        t_start = time.monotonic()

        # Stop/save path does not need live camera frames.
        if self._episode_state.get_state() != self._episode_state.RECORDING:
            return self._finalize_frame(t_start)

        # Proprio is required; images may be blank placeholders if camera is down.
        if self.latest_proprio_msg is None:
            now = time.monotonic()
            last = getattr(self, "_last_waiting_msg_log", 0.0)
            if now - last >= 2.0:
                self._last_waiting_msg_log = now
                self._print_and_say(
                    "Waiting for proprio (g1_debug). "
                    f"image_available={self.latest_image_msg is not None}",
                    say=False,
                )
            return False

        # stream_mode 4 (POSE_PAUSE): skip frame writes during pause; resume
        # immediately when mode returns to POSE (1). Keep polling so save/discard
        # and mode updates still work.
        if self.current_stream_mode == 4:
            return True

        return self._add_data_frame_sonic(t_start)

    def _add_data_frame_sonic(self, t_start: float) -> bool:
        """Build one data frame in Sonic CPP + SMPL mode."""
        assert self.latest_proprio_msg is not None
        proprio = self.latest_proprio_msg

        if self.eef == "brainco" and self.brainco_hand is not None:
            # 2-dim Brainco: [thumb_aux, others]; do not feed into Dex3 7-DoF robot_model.
            left_hand_q, right_hand_q = self.brainco_hand.get_2d_states()
            left_hand_action, right_hand_action = left_hand_q, right_hand_q
            body_q = np.asarray(proprio["body_q"], dtype=np.float64).reshape(-1)
            body_action = np.asarray(proprio["last_action"], dtype=np.float64).reshape(-1)
            whole_q = np.concatenate(
                [body_q, left_hand_q.astype(np.float64), right_hand_q.astype(np.float64)]
            )
            whole_action_wbc = np.concatenate(
                [
                    body_action,
                    left_hand_action.astype(np.float64),
                    right_hand_action.astype(np.float64),
                ]
            )
            # FK for wrist pose only needs body joints; pad Dex3 hand slots with zeros.
            fk_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q,
                left_hand_actuated_joint_values=np.zeros(7, dtype=np.float64),
                right_hand_actuated_joint_values=np.zeros(7, dtype=np.float64),
            )
        elif self.eef == "dex1":
            # Dex1 gripper: 1-dim per hand; FK still uses only body joints.
            body_q = np.asarray(proprio["body_q"], dtype=np.float64).reshape(-1)
            body_action = np.asarray(proprio["last_action"], dtype=np.float64).reshape(-1)
            left_hand_q = np.asarray(proprio.get("left_hand_q", [0.0]), dtype=np.float64).reshape(-1)[:1]
            right_hand_q = np.asarray(proprio.get("right_hand_q", [0.0]), dtype=np.float64).reshape(-1)[:1]
            left_hand_action = np.asarray(
                proprio.get("last_left_hand_action", left_hand_q), dtype=np.float64
            ).reshape(-1)[:1]
            right_hand_action = np.asarray(
                proprio.get("last_right_hand_action", right_hand_q), dtype=np.float64
            ).reshape(-1)[:1]
            whole_q = np.concatenate([body_q, left_hand_q, right_hand_q])
            whole_action_wbc = np.concatenate([body_action, left_hand_action, right_hand_action])
            fk_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q,
                left_hand_actuated_joint_values=np.zeros(7, dtype=np.float64),
                right_hand_actuated_joint_values=np.zeros(7, dtype=np.float64),
            )
        else:
            left_hand_q = proprio["left_hand_q"]
            right_hand_q = proprio["right_hand_q"]
            left_hand_action = proprio["last_left_hand_action"]
            right_hand_action = proprio["last_right_hand_action"]
            whole_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=proprio["body_q"],
                left_hand_actuated_joint_values=left_hand_q,
                right_hand_actuated_joint_values=right_hand_q,
            )
            whole_action_wbc = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=proprio["last_action"],
                left_hand_actuated_joint_values=left_hand_action,
                right_hand_actuated_joint_values=right_hand_action,
            )
            fk_q = whole_q

        self.robot_model.cache_forward_kinematics(fk_q)
        eef_parts = []
        for side in ["left", "right"]:
            placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            pos = placement.translation[:3]
            quat = R.from_matrix(placement.rotation).as_quat(scalar_first=True)
            eef_parts.append(np.concatenate([pos, quat]))
        observation_eef_state = np.concatenate(eef_parts)

        frame_data: dict = {
            "observation.state": whole_q,
            "observation.eef_state": observation_eef_state,
            "action.wbc": whole_action_wbc,
        }

        self._add_cpp_state_features(frame_data, proprio)

        sonic_latency_ms = self._add_sonic_pose_features(frame_data)

        self._add_images_to_frame_data(frame_data)

        self._log_latency_periodic(sonic_latency_ms)

        self.data_exporter.add_frame(frame_data)
        return self._finalize_frame(t_start)

    def _add_cpp_state_features(self, frame_data: dict, proprio: dict) -> None:
        if "base_quat" in proprio:
            base_quat = np.asarray(proprio["base_quat"], dtype=np.float64)
            frame_data["observation.root_orientation"] = base_quat
            frame_data["observation.projected_gravity"] = compute_projected_gravity(
                base_quat
            ).astype(np.float64)

            if "init_ref_data_root_rot_array" in proprio:
                frame_data["observation.cpp_rotation_offset"] = np.asarray(
                    proprio["init_ref_data_root_rot_array"], dtype=np.float64
                )
            else:
                frame_data["observation.cpp_rotation_offset"] = np.array(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
                )
        else:
            frame_data["observation.root_orientation"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            frame_data["observation.projected_gravity"] = np.array(
                [0.0, 0.0, -1.0], dtype=np.float64
            )
            frame_data["observation.cpp_rotation_offset"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        if "init_base_quat" in proprio:
            frame_data["observation.init_base_quat"] = np.asarray(
                proprio["init_base_quat"], dtype=np.float64
            )
        else:
            frame_data["observation.init_base_quat"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        if "delta_heading" in proprio:
            dh = proprio["delta_heading"]
            if isinstance(dh, np.ndarray):
                dh = dh.item() if dh.size == 1 else dh[0]
            frame_data["teleop.delta_heading"] = np.array([float(dh)], dtype=np.float64)
        else:
            frame_data["teleop.delta_heading"] = np.zeros(1, dtype=np.float64)

        if "token_state" in proprio:
            frame_data["action.motion_token"] = np.asarray(proprio["token_state"], dtype=np.float64)
        else:
            frame_data["action.motion_token"] = np.zeros(64, dtype=np.float64)

    def _add_sonic_pose_features(self, frame_data: dict) -> float | None:
        """Add teleop features based on current stream mode."""
        sonic_latency_ms = None

        frame_data["teleop.stream_mode"] = np.array([self.current_stream_mode], dtype=np.int32)

        smpl_msg = self.latest_sonic_msg
        use_smpl = False
        if self.current_stream_mode in (1, 4) and smpl_msg is not None:
            receive_ts = smpl_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = time.time() - receive_ts
                sonic_latency_ms = age_sec * 1000
                self.sonic_timing_monitor.log_time_delta(age_sec)
                if sonic_latency_ms <= 100.0:
                    use_smpl = True
                elif (self.sonic_timing_monitor.failure_count + 1) % 10 == 0:
                    self._print_and_say(
                        f"Sonic pose stale ({sonic_latency_ms:.1f}ms old), using zeros",
                        say=False,
                    )
            else:
                use_smpl = True

        planner_msg = self.latest_planner_msg
        use_planner = False
        if self.current_stream_mode == 5 and planner_msg is not None:
            receive_ts = planner_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = time.time() - receive_ts
                planner_latency_ms = age_sec * 1000
                if sonic_latency_ms is None:
                    sonic_latency_ms = planner_latency_ms
                if planner_latency_ms <= 200.0:
                    use_planner = True
            else:
                use_planner = True

        # SMPL features
        if use_smpl and smpl_msg.get("smpl_joints") is not None:
            joints = np.asarray(smpl_msg["smpl_joints"], dtype=np.float32)
            if joints.ndim == 2:
                joints = joints.flatten()
            frame_data["teleop.smpl_joints"] = np.ascontiguousarray(joints, dtype=np.float32)
        else:
            frame_data["teleop.smpl_joints"] = np.zeros(72, dtype=np.float32)

        if use_smpl and smpl_msg.get("smpl_pose") is not None:
            pose = np.asarray(smpl_msg["smpl_pose"], dtype=np.float32)
            if pose.ndim > 1:
                pose = pose.flatten()
            frame_data["teleop.smpl_pose"] = np.ascontiguousarray(pose, dtype=np.float32)
        else:
            frame_data["teleop.smpl_pose"] = np.zeros(63, dtype=np.float32)

        if use_smpl and smpl_msg.get("body_quat_w") is not None:
            body_quat_w = smpl_msg["body_quat_w"].astype(np.float32)
            frame_data["teleop.body_quat_w"] = body_quat_w
            frame_data["teleop.target_body_orientation"] = self._compute_target_body_orientation(
                body_quat_w, frame_data
            )
        else:
            frame_data["teleop.body_quat_w"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            frame_data["teleop.target_body_orientation"] = quat_to_rot6d(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            )

        frame_data["teleop.left_wrist_joints"] = (
            smpl_msg["left_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("left_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.right_wrist_joints"] = (
            smpl_msg["right_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("right_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )

        frame_data["teleop.smpl_frame_index"] = (
            smpl_msg["frame_index"].astype(np.int64)
            if use_smpl and smpl_msg is not None and smpl_msg.get("frame_index") is not None
            else np.array([0], dtype=np.int64)
        )

        hand_msg = (
            smpl_msg if self.current_stream_mode in (1, 4) and smpl_msg is not None
            else planner_msg if planner_msg is not None
            else smpl_msg
        )
        if self.eef == "brainco" and self.brainco_hand is not None:
            left_2d, right_2d = self.brainco_hand.get_2d_states()
            frame_data["teleop.left_hand_joints"] = left_2d.astype(np.float32)
            frame_data["teleop.right_hand_joints"] = right_2d.astype(np.float32)
        else:
            frame_data["teleop.left_hand_joints"] = (
                hand_msg["left_hand_joints"].astype(np.float32)
                if hand_msg is not None
                and hand_msg.get("left_hand_joints") is not None
                else np.zeros(self.hand_dim, dtype=np.float32)
            )
            frame_data["teleop.right_hand_joints"] = (
                hand_msg["right_hand_joints"].astype(np.float32)
                if hand_msg is not None
                and hand_msg.get("right_hand_joints") is not None
                else np.zeros(self.hand_dim, dtype=np.float32)
            )

        # Planner command fields
        frame_data["teleop.planner_mode"] = np.array(
            [planner_msg["planner_mode"]] if use_planner else [0],
            dtype=np.int32,
        )
        frame_data["teleop.planner_movement"] = (
            planner_msg["planner_movement"].copy()
            if use_planner and planner_msg.get("planner_movement") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.planner_facing"] = (
            planner_msg["planner_facing"].copy()
            if use_planner and planner_msg.get("planner_facing") is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        frame_data["teleop.planner_speed"] = np.array(
            [planner_msg["planner_speed"]] if use_planner else [-1.0],
            dtype=np.float32,
        )
        frame_data["teleop.planner_height"] = np.array(
            [planner_msg["planner_height"]] if use_planner else [-1.0],
            dtype=np.float32,
        )

        # VR 3-point pose
        frame_data["teleop.vr_3pt_position"] = (
            planner_msg["vr_3pt_position"].astype(np.float32)
            if use_planner and planner_msg.get("vr_3pt_position") is not None
            else np.zeros(9, dtype=np.float32)
        )
        if use_planner and planner_msg.get("vr_3pt_orientation") is not None:
            frame_data["teleop.vr_3pt_orientation"] = quat_to_rot6d(
                planner_msg["vr_3pt_orientation"].astype(np.float32)
            )
        else:
            frame_data["teleop.vr_3pt_orientation"] = np.zeros(18, dtype=np.float32)

        return sonic_latency_ms

    def _compute_target_body_orientation(
        self, body_quat_w: np.ndarray, frame_data: dict
    ) -> np.ndarray:
        """Compute yaw-normalised target body orientation as rot6d (6-dim)."""
        delta_heading = float(frame_data.get("teleop.delta_heading", [0.0])[0])

        body_rot = R.from_quat(body_quat_w, scalar_first=True)
        target_rot = R.from_euler("z", delta_heading, degrees=False) * body_rot

        euler = target_rot.as_euler("ZYX", degrees=False)
        current_yaw = euler[0]

        if self._initial_yaw is None:
            self._initial_yaw = current_yaw

        normalised_euler = np.array([current_yaw - self._initial_yaw, euler[1], euler[2]])
        target_quat = (
            R.from_euler("ZYX", normalised_euler, degrees=False)
            .as_quat(scalar_first=True)
            .astype(np.float32)
        )
        return quat_to_rot6d(target_quat)

    def save_and_cleanup(self):
        try:
            self._print_and_say("saving episode done", blocking=False)
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode()
            self._print_and_say(
                f"Recording complete: {self.data_exporter.meta.root}", say=False, blocking=True
            )
        except Exception as e:
            self._print_and_say(f"Error saving episode: {e}", blocking=True)

        try:
            self._state_subscriber.close()
        except Exception:
            pass
        for sock in [self._manager_zmq_socket, self._sonic_zmq_socket]:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        for ctx in [self._sonic_zmq_ctx]:
            if ctx is not None:
                try:
                    ctx.term()
                except Exception:
                    pass

        self._print_and_say("Shutting down data exporter...", say=False)

    def run(self):
        try:
            while True:
                t_start = time.monotonic()
                with self.telemetry.timer("total_loop"):
                    with self.telemetry.timer("poll_state"):
                        self._poll_state_zmq()

                    with self.telemetry.timer("poll_sonic"):
                        self._poll_sonic_zmq_messages()

                    with self.telemetry.timer("poll_image"):
                        img_msg = self._image_subscriber.read()
                        if img_msg is not None:
                            self.latest_image_msg = img_msg
                    # 帧写入
                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                    with self.telemetry.timer("check_recording_commands"):
                        self._check_recording_commands()

                    end_time = time.monotonic()

                elapsed = time.monotonic() - t_start
                sleep_time = self.loop_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if (end_time - t_start) > self.loop_period:
                    self.telemetry.log_timing_info(
                        context="Data Exporter Loop Missed", threshold=0.001
                    )

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()

        finally:
            self.save_and_cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config: SonicDataExporterConfig):
    g1_rm = get_g1_robot_model()

    dataset_features = get_features_sonic_vla(g1_rm, eef=config.eef)
    modality_config = get_modality_config_sonic_vla(g1_rm, eef=config.eef)

    if config.record_wrist_cameras:
        print("[Camera] Wrist cameras enabled — adding to dataset schema")
        dataset_features.update(get_wrist_camera_features())
        wrist_modality = get_wrist_camera_modality_config()
        for key, value in wrist_modality.items():
            if key in modality_config:
                modality_config[key].update(value)
            else:
                modality_config[key] = value

    if config.record_stereo_ego:
        print("[Camera] Head stereo enabled — ego_view_left / ego_view_right")
        # Replace mono ego_view with left/right stereo keys.
        dataset_features.pop("observation.images.ego_view", None)
        dataset_features.update(get_stereo_ego_camera_features())
        modality_config.get("video", {}).pop("ego_view", None)
        stereo_modality = get_stereo_ego_camera_modality_config()
        for key, value in stereo_modality.items():
            if key in modality_config:
                modality_config[key].update(value)
            else:
                modality_config[key] = value

    text_to_speech = TextToSpeech() if config.text_to_speech else None

    robot_config = poll_robot_config_zmq(
        config.state_zmq_host, config.state_zmq_port, config.robot_config_timeout
    )

    if config.dataset_name:
        save_root = f"{config.root_output_dir}/{config.task_name}/{config.dataset_name}"
    else:
        save_root = f"{config.root_output_dir}/{config.task_name}"

    data_exporter = Gr00tDataExporter.create(
        save_root=save_root,
        fps=config.data_collection_frequency,
        features=dataset_features,
        modality_config=modality_config,
        task=config.task_prompt,
        script_config={**robot_config, "record_wrist_cameras": config.record_wrist_cameras, "record_stereo_ego": config.record_stereo_ego},
    )

    data_collector = GrootDataCollector(
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=g1_rm,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
        sonic_data_zmq_host=config.sonic_zmq_host,
        sonic_data_zmq_port=config.sonic_zmq_port,
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=config.state_zmq_port,
        robot_tts=config.robot_tts,
        dds_interface=config.dds_interface,
        eef=config.eef,
    )
    data_collector.run()


if __name__ == "__main__":
    config = tyro.cli(SonicDataExporterConfig)
    main(config)
