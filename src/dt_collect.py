import numpy as np
from datasets import Dataset, DatasetDict
from stable_baselines3 import PPO
from tqdm import tqdm

from MazeEnv import MazeEnv

if __name__ == '__main__':
    env = MazeEnv(render=False)
    model = PPO.load('models/PPO/1000000', env)
    num_episodes = 100
    episodes = []

    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False

        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }

        while not done:
            episode_data['observations'].append(obs.flatten())

            action, _ = model.predict(obs)
            episode_data['actions'].append(np.eye(env.action_space.n)[action])

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_data['rewards'].append(float(reward))
            episode_data['dones'].append(bool(done))

        episodes.append(episode_data)

    data = {
        'observations': [episode['observations'] for episode in episodes],
        'actions': [episode['actions'] for episode in episodes],
        'rewards': [episode['rewards'] for episode in episodes],
        'dones': [episode['dones'] for episode in episodes]
    }
    dataset = Dataset.from_dict(data)
    dataset = DatasetDict({'train': dataset})
    dataset.save_to_disk('episode_data/maze_dataset')

    env.close()