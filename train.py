import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv
from agent import QLearningAgent

def train(episodes=500):
    env = GridWorldEnv()
    agent = QLearningAgent(n_states=env.observation_space.n,
                           n_actions=env.action_space.n)

    rewards_per_episode = []
    success_per_episode = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if truncated:
                break
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        success_per_episode.append(1 if env.state == env.goal else 0)

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_per_episode[-50:])
            succ = np.mean(success_per_episode[-50:]) * 100
            print(f"Episode {ep+1}: avg reward {avg:.2f}, success rate {succ:.0f}%, epsilon {agent.epsilon:.3f}")

    agent.save("q_table.pkl")

    # Plot training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(rewards_per_episode)
    ax1.set_title("Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")

    # Moving average success
    window = 20
    ma = np.convolve(success_per_episode, np.ones(window)/window, mode='valid')
    ax2.plot(ma)
    ax2.set_title("Success Rate (moving avg)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=100)
    print("Saved training_curve.png")

if __name__ == "__main__":
    train()