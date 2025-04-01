import os
import pickle

import numpy as np
from tqdm import tqdm

import wandb
from MazeEnv import MazeEnv

eval_episodes = 100
num_episodes = 1000000
save_dir = 'models/Q'


def save_q_table(q_table):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, 'q_table.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)


def get_player_pos(obs):
    y, x = np.argwhere(obs == 3)[0]
    return x * obs.shape[0] + y


def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(Q[state]))
    else:
        return np.argmax(Q[state])


# alpha: learning rate, aka step size; gamma: discount factor
def q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Q-Table, a value for each state/action pair
    observation_space_size = env.observation_space.shape[0] * \
        env.observation_space.shape[1]
    Q = np.zeros([observation_space_size, env.action_space.n])

    pbar = tqdm(total=num_episodes, dynamic_ncols=True)  # progress bar
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = get_player_pos(state)
        done = False
        episode_reward = 0

        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = get_player_pos(next_state)
            done = terminated or truncated
            # This is what make Q-learning different from SARSA
            best_next_action = np.argmax(Q[next_state, :])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            state = next_state
            episode_reward += reward
        pbar.update(1)

        if episode % 100 == 0:
            log_num = episode // 100 * 50
            avg_reward, avg_length = evaluate_policy(env, Q, eval_episodes)
            pbar.set_description(
                f"\nAverage reward after {episode} episodes, evaluated in {eval_episodes} episodes: {avg_reward:.2f}")
            wandb.log(
                {"rollout/ep_len_mean": avg_length, "rollout/ep_rew_mean": avg_reward}, step=log_num)

    pbar.close()
    return Q


def evaluate_policy(env, Q, num_episodes):
    total_reward = 0
    total_length = 0
    policy = np.argmax(Q, axis=1)

    for _ in range(num_episodes):
        observation, _ = env.reset()
        observation = get_player_pos(observation)
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action = policy[observation]
            observation, reward, terminated, truncated, _ = env.step(action)
            observation = get_player_pos(observation)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        total_reward += episode_reward
        total_length += episode_length

    return total_reward / num_episodes, total_length / num_episodes


if __name__ == '__main__':
    wandb.init(
        project="Maze-rl-Alg-examples",
    )

    env = MazeEnv(render=False)
    Q = q_learning(env, num_episodes)
    save_q_table(Q)