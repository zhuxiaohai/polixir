import torch
import numpy as np
import pickle as pk
from gym import Env
from gym.utils.seeding import np_random
from gym.spaces import Box, MultiDiscrete
import user_states


class VirtualMarketEnv(Env):
    """A very simple example of virtual marketing environment
    """

    MAX_ENV_STEP = 14 # Number of test days in the current phase
    DAY_COUPON_NUM_LIST = [0, 1, 2, 3, 4, 5]
    DISCOUNT_COUPON_LIST = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
    ROI_THRESHOLD = 9.0
    # In real validation environment, if we do not send any coupons in 14 days, we can get this gmv value
    ZERO_GMV = 81840.0763705537

    def __init__(self,
                 initial_user_states: np.ndarray,
                 venv_model: object,
                 seed_number: int = 0):
        """
        Args:
            initial_user_states: The initial states set from the user states of every day
            venv_model: The virtual environment model is trained by default with the revive algorithm package
            act_num_size: The size of the action for each dimension
            obs_size: The size of the community state
            device: Computation device
            seed_number: Random seed
        """
        self.rng = self.seed(seed_number)
        self.initial_user_states = initial_user_states
        self.venv_model = venv_model
        self.current_env_step = None
        self.states = None
        self.done = None
        self._set_action_space([len(VirtualMarketEnv.DAY_COUPON_NUM_LIST), len(VirtualMarketEnv.DISCOUNT_COUPON_LIST)])
        self._set_observation_space(user_states.states_to_observation(initial_user_states[0]).shape)
        self.total_cost, self.total_gmv = None, None

    def seed(self, seed_number):
        return np_random(seed_number)[0]

    def _set_action_space(self, num_list): # discrete platform action
        self.action_space = MultiDiscrete(num_list)

    def _set_observation_space(self, obs_shape, low=0, high=100):
        self.observation_space = Box(low=low, high=high, shape=obs_shape, dtype=np.float32)

    def step(self, action):
        # Convert action from Gym action space to Revive's actual action space
        coupon_num, coupon_discount = action[0], VirtualMarketEnv.DISCOUNT_COUPON_LIST[action[1]]
        # For a fair promotion policy, replicate a single coupon action to all users
        coupon_num, coupon_discount = np.full(self.states.shape[0], coupon_num), np.full(self.states.shape[0], coupon_discount)
        coupon_actions = np.column_stack((coupon_num, coupon_discount))
        # Feed the state and the same coupon actions to decision graph, to fetch the user's response actions
        venv_infer_result = self.venv_model.infer_one_step({"state": self.states, "action_1": coupon_actions})
        user_actions = venv_infer_result["action_2"]
        day_order_num, day_average_fee = user_actions[..., 0], user_actions[..., 1]
        # Compute next states
        self.states = user_states.get_next_state_numpy(self.states, day_order_num, day_average_fee, coupon_num, coupon_discount)
        info = {
            "CouponNum": coupon_num[0],
            "CouponDiscount": coupon_discount[0],
            "UserAvgOrders": day_order_num.round().mean(),
            "UserAvgFee": day_average_fee.mean(),
            "NonZeroOrderCount": np.count_nonzero(day_order_num.round())
        }
        # Compute reward related variables
        day_coupon_used_num = np.minimum(day_order_num, coupon_num)
        day_cost = (1 - coupon_discount) * day_coupon_used_num * day_average_fee
        day_gmv = day_average_fee * day_order_num - day_cost
        day_total_cost = np.sum(day_cost)
        day_total_gmv = np.sum(day_gmv)
        self.total_gmv += day_total_gmv
        self.total_cost += day_total_cost
        # Compute rewards
        if (self.current_env_step+1) < VirtualMarketEnv.MAX_ENV_STEP:
            reward = 0
        else:
            avg_roi = self.total_gmv / max(self.total_cost, 1)
            if avg_roi >= VirtualMarketEnv.ROI_THRESHOLD:
                reward = self.total_gmv / VirtualMarketEnv.ZERO_GMV
            else:
                reward = avg_roi - VirtualMarketEnv.ROI_THRESHOLD
            info["TotalGMV"] = self.total_gmv
            info["TotalROI"] = avg_roi

        self.done = ((self.current_env_step + 1) == VirtualMarketEnv.MAX_ENV_STEP)
        self.current_env_step += 1
        day_total_order_num = int(np.sum(day_order_num))
        day_roi = day_total_gmv / max(day_total_cost, 1)
        return user_states.states_to_observation(self.states, day_total_order_num, day_roi), reward, self.done, info


    def reset(self):
        """Reset the initial states of all users
        Return:
            The group state
        """
        self.states = self.initial_user_states[self.rng.randint(0, self.initial_user_states.shape[0])]
        self.done = False
        self.current_env_step = 0
        self.total_cost, self.total_gmv = 0.0, 0.0
        return user_states.states_to_observation(self.states)


def get_env_instance(states_path, venv_model_path, device = torch.device('cpu')):
    import sys
    import os
    initial_states = np.load(states_path)
    with open(venv_model_path, 'rb') as f:
        sys.path.insert(0, os.path.dirname(os.path.realpath(venv_model_path)))
        venv_model = pk.load(f, encoding='utf-8')
        venv_model.to(device)
    return VirtualMarketEnv(initial_states, venv_model, device=device)
