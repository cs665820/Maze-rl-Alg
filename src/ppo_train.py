import os

import wandb
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback

from MazeEnv import MazeEnv

wandb.init(
    project="Maze-rl-Alg-examples",
    sync_tensorboard=True,
)

models_dir = 'models/PPO'
log_dir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = MazeEnv(render=False)
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000

for i in range(1, 101):
    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name='PPO',
        callback=WandbCallback()
    )
    model.save(f'{models_dir}/{i * TIMESTEPS}')
    restart_timesteps = False

env.close()