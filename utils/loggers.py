from concurrent.futures import ThreadPoolExecutor

import os
import pickle
import random
import numpy as np

from collections import deque

class Logger:
    def __init__(self, log_file):

        self._log_file = open(log_file, 'ab')
        # we log the data in a multithreaded fashion
        self._multithreaded_recording = ThreadPoolExecutor(4)
        self._recording = []

    def log(self, observation, action, reward, done, info):
        self._recording.append({
            'step': [
                observation,
                action,
            ]
        })

    def clear_logs(self):
        self._recording = []

    def on_episode_done(self):
        self._multithreaded_recording.submit(self._commit)

    def _commit(self):
        # we use pickle to store our data
        pickle.dump(self._recording, self._log_file)
        self._log_file.flush()
        del self._recording[:]

    def close(self):
        self._multithreaded_recording.shutdown()
        self._log_file.close()
        #os.chmod(self._log_file.name, 0o444)  # make file read-only after finishing


class Reader:

    def __init__(self, log_file):
        self._log_file = open(log_file, 'rb')

    def read(self):
        end = False
        observations = []
        actions = []

        while not end:
            try:
                log = pickle.load(self._log_file)
                for entry in log:
                    step = entry['step']
                    observations.append(step[0])
                    actions.append(step[1])
            except EOFError:
                end = True

        return observations, actions

    def close(self):
        self._log_file.close()


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_trajectories = 0
        self.buffer = deque()

    def get_sample(self, sample_size):
        if self.num_trajectories < sample_size:
            return random.sample(self.buffer, self.num_trajectories)
        else:
            return random.sample(self.buffer, sample_size)

    def size(self):
        return self.buffer_size

    def add(self, data):
        if self.num_trajectories < self.buffer_size:
            self.buffer.append(data)
            self.num_trajectories += 1
        else:
            self.buffer.popleft()
            self.buffer.append(data)

    def count(self):
        return self.num_trajectories

    def erase(self):
        self.buffer = deque()
        self.num_trajectories = 0
