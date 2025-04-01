import os

from stable_baselines3 import A2C
from wandb.integration.sb3 import WandbCallback

import wandb
from MazeEnv import MazeEnv

wandb.init(
    project="maze-rl-examples",
    sync_tensorboard=True,
)

models_dir = 'models/A2C'
log_dir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = MazeEnv(render=False)
env.reset()

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000

for i in range(1, 101):
    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name='A2C',
        callback=WandbCallback()
    )
    model.save(f'{models_dir}/{i * TIMESTEPS}')
    restart_timesteps = False

env.close()