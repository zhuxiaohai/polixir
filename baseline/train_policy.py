import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from virtual_env import get_env_instance


if __name__ == '__main__':
    save_path = sys.argv[1] if len(sys.argv) > 1 else 'model_checkpoints'
    env = get_env_instance('user_states_by_day.npy', 'venv.pkl')
    model = PPO("MlpPolicy", env, n_steps=840, batch_size=420, verbose=1, tensorboard_log='logs')
    checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path=save_path)
    model.learn(total_timesteps=int(8e6), callback=[checkpoint_callback])
