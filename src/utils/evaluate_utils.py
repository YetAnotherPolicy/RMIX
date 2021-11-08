import os
import json
import pickle

import numpy as np


class EvaluatorUtils(object):
    """
    Save transition data and logits to disk
    """
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.current_episode = 0
        self.save_dir = os.path.join(args.local_results_path, 'transits_logits')
        self.logits_list = []
        self.data = dict()
        os.makedirs(self.save_dir, exist_ok=True)

    def add_logitis(self, logits):
        self.logits_list.append(logits)

    def add_data(self, pre_data, post_data):
        for data in [pre_data, post_data]:
            for k, v in data.items():
                if k not in self.data:
                    self.data[k] = [v]
                else:
                    self.data[k].append(v)

    def save_episode_data(self):
        """
        save episode data (logits and transition)
        batch_data: all transition data
        """
        logits_path = os.path.join(self.save_dir, f'logits_{self.current_episode}.pkl')
        with open(logits_path, 'wb') as f:
            pickle.dump(self.logits_list, f)

        transit_path = os.path.join(self.save_dir, f'transits_{self.current_episode}.json')
        with open(transit_path, 'w') as f:

            for k, v in self.data.items():
                self.data[k] = [np.array(_v).tolist() if isinstance(_v, list) else _v for _v in v]

            json.dump(self.data, f)

        self.logger.console_logger.info(f"Successfully save {logits_path} and {transit_path}")

    def increase_episode_index(self):
        self.current_episode += 1
        self.logits_list = []
        self.data = dict()
