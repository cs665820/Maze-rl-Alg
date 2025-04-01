import argparse
import os

import imageio
import numpy as np
import pygame
import torch
from datasets import DatasetDict

from DataCollector import DecisionTransformerGymDataCollator
from MazeEnv import MazeEnv
from TrainableDT import TrainableDT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the environment')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of episodes to evaluate')
    parser.add_argument('--record', action='store_true', default=False,
                        help='Record the evaluation episodes')
    return parser.parse_args()


def get_action(model, states, actions, returns_to_go, timesteps):
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length:]
    actions = actions[:, -model.config.max_length:]
    returns_to_go = returns_to_go[:, -model.config.max_length:]
    timesteps = timesteps[:, -model.config.max_length:]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat(
        [torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat(
        [torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat(
        [torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat(
        [torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat(
        [torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    action_pred = model.original_forward(
        states=states,
        actions=actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_pred


if __name__ == "__main__":
    args = parse_args()
    dataset = DatasetDict.load_from_disk('episode_data/maze_dataset')
    model = TrainableDT.from_pretrained('models/DT/maze_dt/checkpoint-240')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)
    collector = DecisionTransformerGymDataCollator(dataset["train"])
    env = MazeEnv(render=args.render)
    TARGET_RETURN = 0.8

    state_mean = collector.state_mean.astype(np.float32)
    state_std = collector.state_std.astype(np.float32)

    state_dim = collector.state_dim
    act_dim = collector.act_dim

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    num_episodes = args.num_episodes
    cur_episode = 0
    total_return = 0
    total_length = 0

    for _ in range(num_episodes):
        episode_return, episode_length = 0, 0
        state, _ = env.reset()
        state = state.flatten()
        target_return = torch.tensor(
            TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
        states = torch.from_numpy(state).reshape(
            1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        timesteps = torch.tensor(
            0, device=device, dtype=torch.long).reshape(1, 1)
        frames = []

        for t in range(collector.max_ep_len):
            env.render()
            actions = torch.cat([actions, torch.zeros(
                (1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = get_action(
                model,
                (states - state_mean) / state_std,
                actions,
                target_return,
                timesteps,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, terminated, truncated, _ = env.step(
                np.argmax(action))

            if args.render and args.record:
                frame = pygame.surfarray.array3d(pygame.display.get_surface())
                frame = np.rot90(frame)
                frame = np.flipud(frame)
                frames.append(frame)

            state = state.flatten()
            done = terminated or truncated

            cur_state = torch.from_numpy(state).to(
                device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0, -1] - reward
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones(
                (1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                total_return += episode_return
                total_length += episode_length

                if args.render and args.record:
                    os.makedirs('assets', exist_ok=True)
                    imageio.mimsave(f'assets/dt.gif', frames, fps=10, loop=0)
                break

    avg_return = total_return / num_episodes
    avg_length = total_length / num_episodes
    print(f"Average Return: {avg_return:.2f}")
    print(f"Average Length: {avg_length:.2f}")
    env.close()