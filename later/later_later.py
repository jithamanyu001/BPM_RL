import pandas as pd
import numpy as np
from environment import BusinessLogEnv
from later.later import Actor, Critic
import torch.optim as optim
import torch.nn as nn
import torch


if __name__ == "__main__":
    # Load your data
    data = pd.read_csv("/home/srinivasan/Skan/Models/RL_Framework/filtered_data_reduced.csv")
    case_counts = data.groupby('case_id').size().reset_index(name='event_count')
    case_counts_sorted = case_counts.sort_values(by='event_count', ascending=False)
    top_10_cases = case_counts_sorted.head(10)
    data = data[data['case_id'].isin(top_10_cases['case_id'])]
    columns = ['case_id', 'app_name_activity', 'unhashed_title', 'unhashed_active_url']
    data = data[columns]
    data = data.fillna("")

    # Initialize environment and agent
    k = 2
    env = BusinessLogEnv(data, k)
    state_size = 3 * k  # Each state is encoded as a list of 3 numerical values, and we have 2 states in the sequence
    action_size = len(data['app_name_activity'].unique()) * len(data['unhashed_title'].unique()) * len(data['unhashed_active_url'].unique()) 

    lr_actor = 1e-2
    lr_critic = 3e-2
    gamma = 0.99

    # Initialize Actor and Critic networks
    actor = Actor(state_size, action_size)
    critic = Critic(state_size)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    criterion = nn.MSELoss()

    # Training loop
    max_episodes = 1000
    max_steps = 200

    for episode in range(max_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action with exploration (e.g., noise or epsilon-greedy)
            with torch.no_grad():
                action = actor(torch.FloatTensor(state))
            
            # Execute action in the environment
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            episode_reward += reward
            
            # Compute TD error for Critic
            with torch.no_grad():
                target_value = reward + gamma * critic(torch.FloatTensor(next_state))
            predicted_value = critic(torch.FloatTensor(state))
            td_error = target_value - predicted_value
            
            # Update Critic
            optimizer_critic.zero_grad()
            optimizer_actor.zero_grad()
            critic_loss = criterion(predicted_value[0], target_value[0])
            action_idx = torch.argmax(action, axis=1)
            actor_loss = -td_error * actor(torch.FloatTensor(state))[0][action_idx]
            # print(critic_loss)

            critic_loss.backward(retain_graph = True)
            actor_loss.backward()
            optimizer_critic.step()
            
            # Update Actor using TD error as advantage
            optimizer_actor.step()
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # Print episode information
        print(f"Episode {episode+1}/{max_episodes}, Episode Reward: {episode_reward}")


    # agent = DQNAgent(state_size, action_size)
    
    # episodes = 1000
    # batch_size = 32

    # for e in range(episodes):
    #     state = env.reset()
    #     state = np.reshape(state, [1, state_size])
    #     tot_reward = 0
    #     done = False
    #     for time in range(200):
    #         action = agent.act(state)
    #         # if time%100 ==0:
    #         #     print(action)
    #         next_state, reward, done = env.step(action)
    #         next_state = np.reshape(next_state, [1, state_size])
    #         agent.remember(state, action, reward, next_state, done)
    #         state = next_state
    #         tot_reward+=reward
    #         if done:
    #             print(f"episode: {e}/{episodes}, score: {tot_reward}, e: {agent.epsilon:.2}")
    #             break
    #         if len(agent.memory) > batch_size:
    #             agent.replay(batch_size)
    #     if not done:
    #         print(f"episode: {e}/{episodes}, score: {tot_reward}, e: {agent.epsilon:.2}")
