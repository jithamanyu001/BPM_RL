import numpy as np
from scipy.special import softmax
# epsilon greedy policy here

class QLearningAgent:
    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        print(self.action_size)

    def get_state_key(self, state):
        return tuple(state)

    def get_action(self, state):
        state_key = self.get_state_key(state)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        if state_key not in self.q_table.keys():
            self.q_table[state_key] = np.zeros(self.action_size)
        return np.argmax(self.q_table[state_key])

        # state_key = self.get_state_key(state)
        # if state_key not in self.q_table:
        #     self.q_table[state_key] = np.zeros(self.action_size)
        # q_values = self.q_table[state_key]
        # action_probabilities = softmax(q_values)
        # action = np.random.choice(self.action_size, p=action_probabilities)
        # return action

    def update_q_table(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
