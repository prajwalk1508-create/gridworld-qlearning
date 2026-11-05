import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """
    8x8 GridWorld. Agent starts top-left (0,0), goal is bottom-right (7,7).
    Walls block movement. 4 actions: 0=up, 1=down, 2=left, 3=right.
    """
    def __init__(self, size=8, max_steps=200):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(size * size)

        # Define walls (row, col) - you can change these
        self.walls = {(1, 1), (1, 2), (2, 5), (3, 3), (4, 6), (5, 1), (6, 4), (2, 2)}
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = self.start
        self.steps = 0

    def _to_index(self, pos):
        return pos[0] * self.size + pos[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start
        self.steps = 0
        return self._to_index(self.state), {}

    def step(self, action):
        self.steps += 1
        row, col = self.state

        # Move
        if action == 0:   row -= 1   # up
        elif action == 1: row += 1   # down
        elif action == 2: col -= 1   # left
        elif action == 3: col += 1   # right

        # Boundary check - stay in place if out of bounds
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            row, col = self.state
        # Wall check
        if (row, col) in self.walls:
            row, col = self.state

        self.state = (row, col)

        # Reward
        if self.state == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1   # step penalty (encourages speed)
            done = False

        # Max steps reached
        if self.steps >= self.max_steps:
            done = True

        truncated = self.steps >= self.max_steps and self.state != self.goal
        return self._to_index(self.state), reward, done, truncated, {}

    def render(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        for (r, c) in self.walls:
            grid[r][c] = '#'
        gr, gc = self.goal
        grid[gr][gc] = 'G'
        sr, sc = self.state
        grid[sr][sc] = 'A'
        for row in grid:
            print(' '.join(row))
        print()