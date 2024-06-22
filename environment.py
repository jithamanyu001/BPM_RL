import numpy as np
from collections import deque
import torch

class BusinessLogEnv:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.state = deque(maxlen=k)
        self.case_ids = self.data['case_id'].unique()
        self.current_case = None
        self.current_index = None
        self.activity_name_mapping = {name: idx for idx, name in enumerate(data['app_name_activity'].unique())}
        self.title_mapping = {title: idx for idx, title in enumerate(data['unhashed_title'].unique())}
        self.url_mapping = {url: idx for idx, url in enumerate(data['unhashed_active_url'].unique())}
        self.action_space = self._create_action_space()
        self.reset()
    
    def _create_action_space(self):
        action_space = []
        for activity in self.activity_name_mapping:
            for title in self.title_mapping:
                for url in self.url_mapping:
                    action_space.append((activity, title, url)) # title,
        return action_space

    def reset(self):
        self.current_case = np.random.choice(self.case_ids)
        self.current_index = 0
        self.state.clear()
        case_data = self.data[self.data['case_id'] == self.current_case]
        for i in range(self.k):
            if self.current_index < len(case_data):
                event = case_data.iloc[self.current_index]
                self.state.append(self._encode_event(event))
                self.current_index += 1
        return self._get_state()

    def step(self, action):
        done = False
        reward = -1
        action_idx = torch.argmax(action).item()
        chosen_action = self.action_space[action_idx]

        case_data = self.data[self.data['case_id'] == self.current_case]
        if self.current_index < len(case_data):
            next_event = case_data.iloc[self.current_index]
            if (next_event['app_name_activity'], next_event['unhashed_title'], next_event['unhashed_active_url']) == chosen_action: # , next_event['data_attributes']
                reward = 1
            self.state.append(self._encode_event(next_event))
            self.current_index += 1
        else:
            done = True

        return self._get_state(), reward, done

    def _encode_event(self, event):
        return [
            self.activity_name_mapping[event['app_name_activity']],
            self.title_mapping[event['unhashed_title']],
            self.url_mapping[event['unhashed_active_url']]
        ]

    def _get_state(self):
        return list(self.state)
