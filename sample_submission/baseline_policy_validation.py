import numpy as np
from typing import List, Optional
from stable_baselines3 import PPO

from policy_validation import PolicyValidation
from user_states import get_next_state_numpy, states_to_observation


class BaselinePolicyValidation(PolicyValidation):
    def __init__(self, initial_states_path, policy_model_path):
        self.initial_states = np.load(initial_states_path)
        self.policy_model = PPO.load(policy_model_path)
        self.cur_day_total_order_num = 0
        self.cur_day_roi = 0.0

    def get_next_states(self, cur_states: Optional[np.ndarray], coupon_action: np.ndarray, user_actions: List[np.ndarray]):
        # Unwrap information from parameters
        user_actions = np.array(user_actions)
        day_order_num, day_average_fee = user_actions[..., 0], user_actions[..., 1]
        coupon_num, coupon_discount = np.full(cur_states.shape[0], coupon_action[0]), np.full(cur_states.shape[0], coupon_action[1])
        # Compute next state
        next_states = get_next_state_numpy(cur_states, day_order_num, day_average_fee, coupon_num, coupon_discount)
        # Compute additional information
        day_coupon_used_num = np.minimum(day_order_num, coupon_num)
        day_cost = (1 - coupon_discount) * day_coupon_used_num * day_average_fee
        day_gmv = day_average_fee * day_order_num - day_cost
        day_total_cost = np.sum(day_cost)
        day_total_gmv = np.sum(day_gmv)
        self.cur_day_total_order_num = np.sum(day_order_num)
        self.cur_day_roi = day_total_gmv / max(day_total_cost, 1)
        return next_states

    def get_action_from_policy(self, user_states: Optional[List[np.ndarray]]=None):
        obs = states_to_observation(user_states, self.cur_day_total_order_num, self.cur_day_roi)
        action, _ = self.policy_model.predict(obs, deterministic=True)
        action = action.astype(np.float32)
        action[1] = 0.95 - 0.05 * action[1]
        return action
