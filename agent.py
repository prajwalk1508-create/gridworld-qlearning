import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)   # explore
        return np.argmax(self.q_table[state])          # exploit

    def update(self, state, action, reward, next_state, done):
        best_next = 0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)