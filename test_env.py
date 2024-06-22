import unittest
import pandas as pd
from environment import BusinessLogEnv

class TestBusinessLogEnv(unittest.TestCase):
    def setUp(self):
        data_dict = {
            'case_id': [1, 1, 1, 2, 2, 2],
            'activity_name': ['start', 'middle', 'end', 'start', 'middle', 'end'],
            'title': ['Title1', 'Title2', 'Title3', 'Title1', 'Title2', 'Title3'],
            'url': ['url1', 'url2', 'url3', 'url1', 'url2', 'url3']
        }
        data = pd.DataFrame(data_dict)
        self.env = BusinessLogEnv(data, k=2)
    
    def test_reset(self):
        state = self.env.reset()
        self.assertEqual(len(state), 2)
    
    def test_step(self):
        self.env.reset()
        action = ('middle', 'Title2', 'url2')
        next_state, reward, done = self.env.step(action)
        self.assertEqual(len(next_state), 2)
        self.assertIn(('middle', 'Title2', 'url2'), [(e['activity_name'], e['title'], e['url']) for e in next_state])

if __name__ == "__main__":
    unittest.main()
