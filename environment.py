import numpy as np
from collections import deque
import itertools

class BusinessLogEnv:
    def __init__(self, data, k, pred_vars):
        self.data = data
        self.k = k
        self.pred_vars = pred_vars
        self.state = deque(maxlen=k)
        self.case_ids = self.data['case_id'].unique()
        self.current_case = None
        self.current_index = None
        self.var_maps = {}
        self.var_indices = []
        for pred_var in pred_vars:
            mapping = {name: idx for idx, name in enumerate(data[pred_var].unique())}
            indices = [idx for idx, name in enumerate(data[pred_var].unique())]
            self.var_maps[pred_var] = mapping
            self.var_indices.append(indices)
        self.action_space = list(itertools.product(*self.var_indices))
        # print(self.action_space)

    def reset(self):
        self.current_case = np.random.choice(self.case_ids)
        self.current_index = 0
        self.state.clear()
        # case_data = self.data[self.data['case_id'] == self.current_case]
        for i in range(self.k):
            if self.current_index < len(self.data):
                event = self.data.iloc[self.current_index]
                self.state.append(self.encode_event(event))
                self.current_index += 1
        return tuple(self.state)

    def step(self, action):
        # action is the index of a  tuple (v1, v2, ...) in action_space
        done = False
        reward = -1
        # case_data = self.data[self.data['case_id'] == self.current_case]
        if self.current_index < len(self.data):
            next_event = self.data.iloc[self.current_index]
            next_list = []
            for pred_var in self.pred_vars:
                next_list.append(next_event[pred_var])
            if next_list == self.action_space[action]: 
                reward = 1
            self.state.append(self.encode_event(next_event))
            # print(action)
            self.current_index += 1
        else:
            done = True

        return tuple(self.state), reward, done

    def encode_event(self, event):
        encoded = []
        for pred_var in self.pred_vars:
            encoded.append(self.var_maps[pred_var][event[pred_var]])
        return tuple(encoded)
