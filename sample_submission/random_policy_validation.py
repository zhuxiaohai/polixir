import numpy as np
from typing import List, Optional

from policy_validation import PolicyValidation


class RandomPolicyValidation(PolicyValidation):
    @property
    def initial_states(self):
        return None

    def get_next_states(self, cur_states: None, coupon_action: np.ndarray, user_actions: List[np.ndarray]):
        return None # Using random policy, we do not need the depiction of user

    def get_action_from_policy(self, user_states: Optional[List[np.ndarray]]=None):
        num = np.random.randint(1, 3) # 1, 2, uniformly choose one
        discount = 0.95 - 0.05 * np.random.randint(0, 5) # 0.95, 0.90, 0.85, 0.80, 0.75, uniformly choose one
        return np.array([num, discount])
