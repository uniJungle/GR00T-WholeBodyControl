"""Brainco Revo2 hand wrapper for SONIC teleop (trigger open/close)."""

from __future__ import annotations

from multiprocessing import Array, Lock
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from eef.brainco.brainco_controller import (
    Brainco_Controller,
    MotorCmds_,
    MotorStates_,
    brainco_Num_Motors,
    kTopicbraincoLeftCommand,
    kTopicbraincoLeftState,
    kTopicbraincoRightCommand,
    kTopicbraincoRightState,
)
from unitree_sdk2py.core.channel import ChannelSubscriber


class Brainco:
    """Active or passive Brainco hand interface.

    Active mode (`passive=False`): publish open/close targets on DDS from
    ``set_gripper_targets(left_ratio, right_ratio)`` where 0=open, 1=closed.

    Passive mode: only monitor external DDS cmd/state (for recording).
    """

    def __init__(
        self,
        passive: bool = False,
        network_interface: str | None = None,
        init_channel_factory: bool = True,
    ):
        self.passive = passive
        self.network_interface = network_interface
        self.left_hand_pos_array = Array("d", 6, lock=True)
        self.right_hand_pos_array = Array("d", 6, lock=True)
        self.dual_hand_data_lock = Lock()
        self.dual_hand_state_array_out = Array("d", 6 * 2, lock=False)
        self.dual_hand_action_array_out = Array("d", 6 * 2, lock=False)

        self.l_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.r_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if init_channel_factory:
            if network_interface:
                ChannelFactoryInitialize(0, network_interface)
            else:
                ChannelFactoryInitialize(0)

        if self.passive:
            self.hand_ctrl = None
            self._running = True
            self._init_passive_dds_monitor()
        else:
            self.hand_ctrl = Brainco_Controller(
                self.left_hand_pos_array,
                self.right_hand_pos_array,
                self.dual_hand_data_lock,
                self.dual_hand_state_array_out,
                self.dual_hand_action_array_out,
                simulation_mode=False,
                network_interface=network_interface,
                # ChannelFactory already initialized above.
                init_channel_factory=False,
            )

        self.open_pose_801 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.open_pose_504 = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        self.open_pose = self.open_pose_801.copy()
        self.close_pose = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    def set_gripper_targets(self, left_ratio, right_ratio):
        if self.passive:
            return

        left_ratio = float(np.clip(left_ratio, 0.0, 1.0))
        right_ratio = float(np.clip(right_ratio, 0.0, 1.0))

        # 提取为 2-dim 控制逻辑：
        # dim 0: ThumbAux (index 1)
        # dim 1: 其他 5 个电机 (index 0, 2, 3, 4, 5)
        l_thumb_aux = self.open_pose[1] + (self.close_pose[1] - self.open_pose[1]) * left_ratio
        l_others = self.open_pose[0] + (self.close_pose[0] - self.open_pose[0]) * left_ratio
        
        r_thumb_aux = self.open_pose[1] + (self.close_pose[1] - self.open_pose[1]) * right_ratio
        r_others = self.open_pose[0] + (self.close_pose[0] - self.open_pose[0]) * right_ratio

        self.set_2d_targets([l_thumb_aux, l_others], [r_thumb_aux, r_others])

    def set_2d_targets(self, left_2d, right_2d):
        """
        直接下发 2-dim 控制指令。
        left_2d, right_2d: [thumb_aux_val, others_val]
        """
        if self.passive:
            return

        def to_6d(val_2d):
            cmd = np.zeros(6, dtype=np.float64)
            cmd[1] = val_2d[0]                # ThumbAux
            cmd[[0, 2, 3, 4, 5]] = val_2d[1]  # Thumb, Index, Middle, Ring, Pinky
            return cmd

        l_cmd = to_6d(left_2d)
        r_cmd = to_6d(right_2d)

        with self.left_hand_pos_array.get_lock():
            self.left_hand_pos_array[:] = l_cmd
        with self.right_hand_pos_array.get_lock():
            self.right_hand_pos_array[:] = r_cmd

        self.l_cmd = l_cmd
        self.r_cmd = r_cmd

    def get_2d_states(self):
        """
        获取 2-dim 的手部状态反馈 (基于 6-dim 实测值的均值聚合)。
        返回: left_2d, right_2d
        """
        left_6d, right_6d = self.get_hand_states()

        def to_2d(val_6d):
            arr = np.asarray(val_6d, dtype=np.float64).reshape(-1)
            thumb_aux = arr[1]
            others_mean = float(np.mean(arr[[0, 2, 3, 4, 5]]))
            return np.array([thumb_aux, others_mean], dtype=np.float64)

        return to_2d(left_6d), to_2d(right_6d)

    def set_hand_targets(self, targets):
        if self.passive:
            return

        with self.left_hand_pos_array.get_lock():
            self.left_hand_pos_array[:] = targets[:6]
        with self.right_hand_pos_array.get_lock():
            self.right_hand_pos_array[:] = targets[6:]

        self.l_cmd = targets[:6]
        self.r_cmd = targets[6:]

    def get_hand_states(self):
        with self.dual_hand_data_lock:
            left_ee_state = self.dual_hand_state_array_out[:6]
            right_ee_state = self.dual_hand_state_array_out[6:]
        return left_ee_state, right_ee_state

    def change_open_pose(self, fsm_mode):
        if self.passive:
            return
        if fsm_mode == 801:
            self.open_pose[:] = self.open_pose_801
        elif fsm_mode == 504:
            self.open_pose[:] = self.open_pose_504

    def close(self):
        """Stop DDS threads. Safe to call multiple times."""
        if self.passive:
            self._running = False
            if getattr(self, "passive_thread", None) is not None and self.passive_thread.is_alive():
                self.passive_thread.join(timeout=1.0)
            return

        if self.hand_ctrl is not None:
            self.hand_ctrl.close()
            self.hand_ctrl = None

    def _init_passive_dds_monitor(self):
        self.left_cmd_sub = ChannelSubscriber(kTopicbraincoLeftCommand, MotorCmds_)
        self.left_cmd_sub.Init()
        self.right_cmd_sub = ChannelSubscriber(kTopicbraincoRightCommand, MotorCmds_)
        self.right_cmd_sub.Init()
        self.left_state_sub = ChannelSubscriber(kTopicbraincoLeftState, MotorStates_)
        self.left_state_sub.Init()
        self.right_state_sub = ChannelSubscriber(kTopicbraincoRightState, MotorStates_)
        self.right_state_sub.Init()

        self.passive_thread = threading.Thread(target=self._passive_dds_loop, daemon=True)
        self.passive_thread.start()

    def _passive_dds_loop(self):
        while self._running:
            left_cmd = self.left_cmd_sub.Read()
            right_cmd = self.right_cmd_sub.Read()
            left_state = self.left_state_sub.Read()
            right_state = self.right_state_sub.Read()

            if left_cmd is not None and getattr(left_cmd, "cmds", None):
                self.l_cmd = self._read_motor_values(left_cmd.cmds, "q")
            if right_cmd is not None and getattr(right_cmd, "cmds", None):
                self.r_cmd = self._read_motor_values(right_cmd.cmds, "q")

            with self.dual_hand_data_lock:
                if left_state is not None and getattr(left_state, "states", None):
                    self.dual_hand_state_array_out[:brainco_Num_Motors] = self._read_motor_values(
                        left_state.states, "q"
                    )
                if right_state is not None and getattr(right_state, "states", None):
                    self.dual_hand_state_array_out[brainco_Num_Motors:] = self._read_motor_values(
                        right_state.states, "q"
                    )
                self.dual_hand_action_array_out[:brainco_Num_Motors] = self.l_cmd
                self.dual_hand_action_array_out[brainco_Num_Motors:] = self.r_cmd

            time.sleep(0.002)

    @staticmethod
    def _read_motor_values(items, attr):
        values = np.zeros(brainco_Num_Motors, dtype=np.float64)
        for idx in range(min(brainco_Num_Motors, len(items))):
            values[idx] = float(getattr(items[idx], attr, 0.0))
        return values


if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp4s0")
    hand = Brainco(init_channel_factory=False)
    try:
        hand.change_open_pose(fsm_mode=801)
        hand.set_gripper_targets(0.5, 0.5)
        time.sleep(2)
        hand.set_gripper_targets(0.0, 0.0)
        time.sleep(2)
        while True:
            hand.get_hand_states()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        hand.set_gripper_targets(0.0, 0.0)
        time.sleep(0.2)
        hand.close()
