import gymnasium as gym
import numpy as np
import pygame

MAZE = [
    [3, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 2]
]
CELL_SIZE = 100
MAZE_SIZE = len(MAZE)
TOTAL_WIDTH = CELL_SIZE * MAZE_SIZE


class MazeEnv(gym.Env):
    """
    Custom Maze Environment for reinforcement learning.
    """

    def __init__(self, render=True):
        super(MazeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Box(
            low=0, high=3, shape=(MAZE_SIZE, MAZE_SIZE), dtype=int)
        self.agent_pos = (0, 0)
        self.maze = [row[:] for row in MAZE]
        self.step_count = 0

        self.screen: pygame.Surface | None = None
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_WIDTH))

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        """
        super(MazeEnv, self).reset(seed=seed, options=options)
        self.agent_pos = (0, 0)
        self.maze = [row[:] for row in MAZE]
        self.step_count = 0

        return np.array(self.maze), {}

    def step(self, action):
        """
        Takes a step in the environment based on the action taken.
        """
        self.step_count += 1
        if action < 0 or action >= self.action_space.n:
            raise ValueError("Invalid action")

        x, y = self.agent_pos
        self.maze[y][x] = 0

        movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = movements[action]

        new_x = max(0, min(MAZE_SIZE - 1, x + dx))
        new_y = max(0, min(MAZE_SIZE - 1, y + dy))

        if self.maze[new_y][new_x] != 1:
            x, y = new_x, new_y

        self.agent_pos = (x, y)
        self.maze[y][x] = 3

        done = self.maze[MAZE_SIZE - 1][MAZE_SIZE - 1] == 3
        truncated = self.step_count >= 50 and not done
        reward = 1 if done else -0.01

        return np.array(self.maze), reward, done, truncated, {}

    def render(self):
        """
        Renders the current state of the environment.
        """
        if not self.screen:
            return

        self.screen.fill((255, 255, 255))
        for i in range(MAZE_SIZE):
            for j in range(MAZE_SIZE):
                cell_value = self.maze[i][j]
                if cell_value == 1:
                    color = (0, 0, 0)
                elif cell_value == 2:
                    color = (0, 255, 0)
                else:
                    color = (255, 255, 255)

                pygame.draw.rect(self.screen, color,
                                 (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))

                if cell_value == 3:
                    pygame.draw.circle(
                        self.screen,
                        (0, 0, 255),
                        (j * CELL_SIZE + CELL_SIZE // 2,
                         i * CELL_SIZE + CELL_SIZE // 2),
                        CELL_SIZE // 4
                    )

        pygame.display.flip()
        pygame.time.delay(100)