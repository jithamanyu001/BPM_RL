import pandas as pd
from environment import BusinessLogEnv
from q_learning_agent import QLearningAgent
from sarsa_agent import SarsaAgent
import numpy as np


data = pd.read_csv("D:/Projs/Skan/Files/BPM_RL/filtered_data_reduced.csv").head(10000)
# x - correct   2x - 10000    r + 10000 / 20000
# data = data[data['case_id']=="62ca1700177dda71feb36f7770f30e06/149"]

data = data.fillna("")
pred_vars = ['app_name_activity', 'unhashed_active_url']
lrs=[0.01, 0.005]
eps=[(1, 0.1), (0.1, 0.01), (0.01, 0.001)]
k = 10
env = BusinessLogEnv(data, k, pred_vars)
for lr in lrs:
    for epsilon,epsilon_min in eps:
        print(f"Results for {lr=}, {epsilon=}, {epsilon_min=}")
        k = 10
        env = BusinessLogEnv(data, k, pred_vars)

        action_size = len(env.action_space)
        agent = SarsaAgent(action_size, learning_rate=lr, discount_factor=0.99, epsilon=epsilon, epsilon_decay=0.995, epsilon_min=epsilon_min)

        num_episodes = 10000

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update_q_table(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            if episode%100 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


            # Optionally, save the Q-table every few episodes
            if (episode + 1) % 100 == 0:
                with open(f'Q_tables/q_table_{episode + 1}_{str(lr)}_{str(epsilon)}_{str(epsilon_min)}.npy', 'wb') as f:
                    np.save(f, agent.q_table)
