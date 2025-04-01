import argparse
import os

import imageio
import numpy as np
import pygame
from stable_baselines3 import A2C

from MazeEnv import MazeEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the environment')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of episodes to evaluate')
    parser.add_argument('--record', action='store_true', default=False,
                        help='Record the evaluation episodes')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    env = MazeEnv(render=args.render)
    model = A2C.load('models/A2C/1000000', env)
    episodes = args.num_episodes

    total_return = 0
    total_length = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        episode_return = 0
        episode_length = 0
        frames = []

        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            if args.render and args.record:
                frame = pygame.surfarray.array3d(pygame.display.get_surface())
                frame = np.rot90(frame)
                frame = np.flipud(frame)
                frames.append(frame)

            done = terminated or truncated

            episode_return += reward
            episode_length += 1

        if args.render and args.record:
            os.makedirs('assets', exist_ok=True)
            imageio.mimsave(f'assets/a2c.gif', frames, fps=10, loop=0)

        total_return += episode_return
        total_length += episode_length

    avg_return = total_return / episodes
    avg_length = total_length / episodes
    print(f'Average Return: {avg_return:.2f}')
    print(f'Average Length: {avg_length:.2f}')
    env.close()