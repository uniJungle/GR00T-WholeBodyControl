from eef.dex1.dex_controller import Dex1_1_Gripper_Controller
from multiprocessing import Array, Lock, Value
import numpy as np
class Dex1:
    def __init__(self, network_interface=None):

        self.left_hand_pos_array = Value('d', 1, lock=True)
        self.right_hand_pos_array = Value('d', 1, lock=True)
        self.dual_hand_data_lock = Lock()
        self.dual_hand_state_array_out = Array('d', 2, lock=False)   # 左右手状态
        self.dual_hand_action_array_out = Array('d', 2, lock=False)  # 左右手动作

        self.gripper_ctrl = Dex1_1_Gripper_Controller(
            self.left_hand_pos_array,
            self.right_hand_pos_array,
            self.dual_hand_data_lock,
            self.dual_hand_state_array_out,
            self.dual_hand_action_array_out,
            simulation_mode=False,
            network_interface=network_interface,
        ) 
        self.l_cmd=0.0
        self.r_cmd=0.0

    def set_gripper_ratios(self, left_ratio, right_ratio):
        with self.left_hand_pos_array.get_lock():
            self.left_hand_pos_array.value = left_ratio
        with self.right_hand_pos_array.get_lock():
            self.right_hand_pos_array.value = right_ratio

        self.l_cmd=left_ratio
        self.r_cmd=right_ratio
        
    def get_hand_states(self):
        with self.dual_hand_data_lock:
            left_ee_state = self.dual_hand_state_array_out[:1]
            right_ee_state = self.dual_hand_state_array_out[1:]

        return left_ee_state,right_ee_state
    
    def change_open_pose(self, fsm_mode):
        pass