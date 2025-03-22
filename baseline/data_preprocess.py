import sys
import pandas as pd
import numpy as np
import user_states


def data_preprocess(offline_data_path: str, base_days=30):
    """Define user state from original offline data and generate new offline data containing user state
    Args:
        offline_data_path: Path of original offline data
        base_days: The number of historical days required for defining the initial user state

    Return:
        new_offline_data(pd.DataFrame): New offline data containing user state.
            User state, 3 dimension：Historical total order number、historical average of non-zero day order number、and historical average of non-zero day order fee
        user_states_by_day(np.ndarray): An array of shape (number_of_days, number_of_users, 3), contains all the user states.
            If base_days is equal to 30, the number_of_days is equal to total_days - base_days, 30.
            And user_states_by_day[0] means all the user states in the first day.
        evaluation_start_states(np.ndarray): the states for the first day of validation in this competition
        venv_dataset: New offline data in dictionary form, grouped by states, user actions and coupon actions.
    """
    df = pd.read_csv(offline_data_path)
    total_users = df["index"].max() + 1
    total_days = df["step"].max() + 1
    venv_days = total_days - base_days
    state_dims = user_states.get_state_dims()

    # Prepare processed data
    user_states_by_day    = []
    user_actions_by_day   = []
    coupon_actions_by_day = []
    evaluation_start_states = np.empty((total_users, state_dims))

    # Recurrently unroll states from offline dataset
    states = np.zeros((total_users, state_dims)) # State of day 0 to be all zero
    for current_day, day_user_data in df.groupby("step"):
        # Fetch field data
        day_coupon_num  = day_user_data["day_deliver_coupon_num"].to_numpy()
        coupon_discount = day_user_data["coupon_discount"].to_numpy()
        day_order_num   = day_user_data["day_order_num"].to_numpy()
        day_average_fee = day_user_data["day_average_order_fee"].to_numpy()
        # Unroll next state
        next_states = user_states.get_next_state_numpy(states, day_order_num, day_average_fee, day_coupon_num, coupon_discount)
        # Collect data for new offline dataset
        if current_day >= base_days:
            user_states_by_day.append(states)
            user_actions_by_day.append(np.column_stack((day_order_num, day_average_fee)))
            coupon_actions_by_day.append(np.column_stack((day_coupon_num, coupon_discount)))
        if current_day >= total_days:
            evaluation_start_states = next_states # Evaluation starts from the final state in the offline dataset
        states = next_states

    # Group states by users (by trajectory) and generate processed offline dataset
    venv_dataset = {
        "state": np.swapaxes(user_states_by_day, 0, 1).reshape((-1, state_dims)), # Sort by users (trajectory)
        "action_1": np.swapaxes(coupon_actions_by_day, 0, 1).reshape((-1, 2)),
        "action_2": np.swapaxes(user_actions_by_day, 0, 1).reshape((-1, 2)),
        "index": np.arange(venv_days, (total_users + 1) * venv_days, venv_days) # End index for each trajectory
    }
    traj_indices, step_indices = np.mgrid[0:total_users, 0:venv_days].reshape((2, -1, 1)) # Prepare indices
    new_offline_data = pd.DataFrame(
        data=np.concatenate([traj_indices, venv_dataset["state"], venv_dataset["action_1"], venv_dataset["action_2"], step_indices], -1),
        columns=["index", *user_states.get_state_names(), "day_deliver_coupon_num", "coupon_discount", "day_order_num", "day_average_order_fee", "step"])

    return new_offline_data, np.array(user_states_by_day), evaluation_start_states, venv_dataset


if __name__ == "__main__":
    offline_data = sys.argv[1]
    new_offline_data, user_states_by_day, evaluation_start_states, new_offline_data_dict = data_preprocess(offline_data)
    print(new_offline_data.shape)
    print(user_states_by_day.shape)
    print(evaluation_start_states.shape)
    new_offline_data.to_csv('offline_592_3_dim_state.csv', index=False)
    np.save('user_states_by_day.npy', user_states_by_day)
    np.save('evaluation_start_states.npy', evaluation_start_states)
    np.savez('venv.npz', **new_offline_data_dict)
