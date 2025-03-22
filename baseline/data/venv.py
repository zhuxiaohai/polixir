from typing import Dict
import torch
import user_states


def get_next_state(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    # state    -> User State
    # action_2 -> User action
    # action_1 -> Coupon promotion action, only exists when as ingress node to next_state in decision graph
    return user_states.get_next_state_torch(inputs["state"], inputs["action_2"], inputs.get("action_1"))
