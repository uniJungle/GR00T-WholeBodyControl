"""Unitree DDS controller for Brainco Revo2 hands (6 DoF each)."""

from __future__ import annotations

from enum import IntEnum
from multiprocessing import Array
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

try:
    import logging_mp

    logger_mp = logging_mp.get_logger(__name__)
except ImportError:
    import logging

    logger_mp = logging.getLogger(__name__)

brainco_Num_Motors = 6
kTopicbraincoLeftCommand = "rt/brainco/left/cmd"
kTopicbraincoLeftState = "rt/brainco/left/state"
kTopicbraincoRightCommand = "rt/brainco/right/cmd"
kTopicbraincoRightState = "rt/brainco/right/state"


class ControlMode(IntEnum):
    RETARGET = 0
    DIRECT = 1


class Brainco_Controller:
    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
        control_mode=ControlMode.DIRECT,
        network_interface: str | None = None,
        init_channel_factory: bool = True,
    ):
        logger_mp.info("Initialize Brainco_Controller...")
        self.fps = fps
        self.hand_sub_ready = False
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.control_mode = control_mode
        self.hand_retargeting = None
        self.running = True

        if init_channel_factory:
            if self.simulation_mode:
                ChannelFactoryInitialize(1)
            elif network_interface:
                ChannelFactoryInitialize(0, network_interface)
            else:
                ChannelFactoryInitialize(0)

        self.LeftHandCmb_publisher = ChannelPublisher(kTopicbraincoLeftCommand, MotorCmds_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicbraincoRightCommand, MotorCmds_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicbraincoLeftState, MotorStates_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicbraincoRightState, MotorStates_)
        self.RightHandState_subscriber.Init()

        self.left_hand_state_array = Array("d", brainco_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", brainco_Num_Motors, lock=True)

        self.subscribe_state_thread = threading.Thread(
            target=self._subscribe_hand_state, daemon=True
        )
        self.subscribe_state_thread.start()

        while not self.hand_sub_ready:
            time.sleep(0.1)
            logger_mp.warning("[brainco_Controller] Waiting to subscribe dds...")
        logger_mp.info("[brainco_Controller] Subscribe dds ok.")

        self.hand_control_thread = threading.Thread(
            target=self.control_thread,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
            daemon=True,
        )
        self.hand_control_thread.start()
        logger_mp.info("Initialize brainco_Controller OK!\n")

    def _subscribe_hand_state(self):
        while self.running:
            left_hand_msg = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            self.hand_sub_ready = True
            if left_hand_msg is not None and right_hand_msg is not None:
                for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.states[id].q
                for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.states[id].q
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_hand_msg.cmds[id].q = left_q_target[idx]
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_hand_msg.cmds[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_hand_msg)
        self.RightHandCmb_publisher.Write(self.right_hand_msg)

    def control_thread(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        left_q_target = np.full(brainco_Num_Motors, 0.0)
        right_q_target = np.full(brainco_Num_Motors, 0.0)

        self.left_hand_msg = MotorCmds_()
        self.left_hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Left_Hand_JointIndex))
        ]
        self.right_hand_msg = MotorCmds_()
        self.right_hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Right_Hand_JointIndex))
        ]

        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_hand_msg.cmds[id].q = 0.0
            self.left_hand_msg.cmds[id].dq = 1.0
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_hand_msg.cmds[id].q = 0.0
            self.right_hand_msg.cmds[id].dq = 1.0

        try:
            while self.running:
                start_time = time.time()
                state_data = np.hstack(
                    (self.left_hand_state_array[:].copy(), self.right_hand_state_array[:].copy())
                )

                if self.control_mode == ControlMode.RETARGET:
                    with left_hand_array.get_lock():
                        left_hand_data = np.array(left_hand_array[:]).reshape(25, 3).copy()
                    with right_hand_array.get_lock():
                        right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                    if (
                        not np.all(right_hand_data == 0.0)
                        and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15]))
                        and self.hand_retargeting is not None
                    ):
                        ref_left_value = (
                            left_hand_data[self.hand_retargeting.left_indices[1, :]]
                            - left_hand_data[self.hand_retargeting.left_indices[0, :]]
                        )
                        ref_right_value = (
                            right_hand_data[self.hand_retargeting.right_indices[1, :]]
                            - right_hand_data[self.hand_retargeting.right_indices[0, :]]
                        )

                        left_q_target = self.hand_retargeting.left_retargeting.retarget(
                            ref_left_value
                        )[self.hand_retargeting.left_dex_retargeting_to_hardware]
                        right_q_target = self.hand_retargeting.right_retargeting.retarget(
                            ref_right_value
                        )[self.hand_retargeting.right_dex_retargeting_to_hardware]

                        def normalize(val, min_val, max_val):
                            return 1.0 - np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                        for idx in range(brainco_Num_Motors):
                            if idx == 0:
                                left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.52)
                                right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.52)
                            elif idx == 1:
                                left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.05)
                                right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.05)
                            elif idx >= 2:
                                left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.47)
                                right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.47)
                else:
                    with left_hand_array.get_lock():
                        left_q_target = np.array(left_hand_array[:]).copy()
                    with right_hand_array.get_lock():
                        right_q_target = np.array(right_hand_array[:]).copy()

                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)

                time_elapsed = time.time() - start_time
                sleep_time = max(0.0, (1.0 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("brainco_Controller has been closed.")

    def close(self):
        """Stop control / subscribe threads."""
        self.running = False
        if self.hand_control_thread.is_alive():
            self.hand_control_thread.join(timeout=1.0)
        if self.subscribe_state_thread.is_alive():
            self.subscribe_state_thread.join(timeout=1.0)


class Brainco_Right_Hand_JointIndex(IntEnum):
    kRightHandThumb = 0
    kRightHandThumbAux = 1
    kRightHandIndex = 2
    kRightHandMiddle = 3
    kRightHandRing = 4
    kRightHandPinky = 5


class Brainco_Left_Hand_JointIndex(IntEnum):
    kLeftHandThumb = 0
    kLeftHandThumbAux = 1
    kLeftHandIndex = 2
    kLeftHandMiddle = 3
    kLeftHandRing = 4
    kLeftHandPinky = 5
