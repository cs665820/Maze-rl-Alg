import argparse
import os
import pickle

import imageio
import numpy as np
import pygame

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


def load_q_table():
    filename = 'models/Q/q_table.pkl'
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Q-table file not found: {filename}")

    with open(filename, 'rb') as f:
        q_table = pickle.load(f)

    return q_table


def get_player_pos(obs):
    y, x = np.argwhere(obs == 3)[0]
    return x * obs.shape[0] + y


if __name__ == '__main__':
    args = parse_args()
    env = MazeEnv(render=args.render)
    Q = load_q_table()
    episodes = args.num_episodes

    total_return = 0
    total_length = 0

    for _ in range(episodes):
        state, _ = env.reset()
        state = get_player_pos(state)
        done = False

        episode_return = 0
        episode_length = 0
        frames = []

        while not done:
            env.render()
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)

            if args.render and args.record:
                frame = pygame.surfarray.array3d(pygame.display.get_surface())
                frame = np.rot90(frame)
                frame = np.flipud(frame)
                frames.append(frame)

            state = get_player_pos(state)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

        if args.render and args.record:
            os.makedirs('assets', exist_ok=True)
            imageio.mimsave(f'assets/qlearn.gif', frames, fps=10, loop=0)

        total_return += episode_return
        total_length += episode_length

    avg_return = total_return / episodes
    avg_length = total_length / episodes
    print(f'Average Return: {avg_return:.2f}')
    print(f'Average Length: {avg_length:.2f}')
    env.close()