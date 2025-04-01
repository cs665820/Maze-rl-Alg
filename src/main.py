from MazeEnv import MazeEnv

if __name__ == '__main__':
    env = MazeEnv()

    for _ in range(10):
        state = env.reset()
        done = False

        while not done:
            env.render()
            action = env.action_space.sample()  # Random action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()