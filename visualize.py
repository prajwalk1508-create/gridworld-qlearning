import numpy as np
import matplotlib.pyplot as plt
import imageio
from grid_env import GridWorldEnv
from agent import QLearningAgent

def draw_policy(env, agent, filename="policy.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    arrows = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.walls:
                ax.add_patch(plt.Rectangle((c, env.size-1-r), 1, 1, color='black'))
            elif (r, c) == env.goal:
                ax.add_patch(plt.Rectangle((c, env.size-1-r), 1, 1, color='green'))
            else:
                s = r * env.size + c
                a = np.argmax(agent.q_table[s])
                dx, dy = arrows[a]
                ax.arrow(c+0.5, env.size-1-r+0.5, dx, dy,
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    ax.set_xlim(0, env.size); ax.set_ylim(0, env.size)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Learned Policy")
    plt.savefig(filename, dpi=100)
    print(f"Saved {filename}")

def make_gif(env, agent, filename="agent.gif"):
    state, _ = env.reset()
    frames = []
    done = False
    while not done:
        fig, ax = plt.subplots(figsize=(4, 4))
        for r in range(env.size):
            for c in range(env.size):
                color = 'white'
                if (r, c) in env.walls: color = 'black'
                elif (r, c) == env.goal: color = 'green'
                elif (r, c) == env.state: color = 'red'
                ax.add_patch(plt.Rectangle((c, env.size-1-r), 1, 1, color=color, ec='gray'))
        ax.set_xlim(0, env.size); ax.set_ylim(0, env.size)
        ax.set_xticks([]); ax.set_yticks([])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy()   # drop alpha channel, keep RGB
        frames.append(frame)
        plt.close(fig)

        action = np.argmax(agent.q_table[state])   # greedy, no exploration
        state, reward, done, truncated, _ = env.step(action)
        if truncated: break
    imageio.mimsave(filename, frames, fps=4)
    print(f"Saved {filename}")

if __name__ == "__main__":
    env = GridWorldEnv()
    agent = QLearningAgent(n_states=env.observation_space.n, n_actions=env.action_space.n)
    agent.load("q_table.pkl")
    draw_policy(env, agent)
    make_gif(env, agent)