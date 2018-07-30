import gym
import numpy as np
from collections import deque
import pickle
import os

def action_with_index(index):
    if index == 0:
        return 0
    elif index == 1:
        return 2
    elif index == 2:
        return 3
    else:
        NotImplementedError

def init_env():
    env = gym.make('PongNoFrameskip-v4')
    env.seed(0)
    env = DemoEnv(env, demo_file_name="Pong.demo")
    env = MaxMergeSkipEnv(env, skip=4)
    env = BlackWhiteEnv(env)
    return env

class GymWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GymWrapper, self).__init__(env)

    def decrement_starting_point(self, nr_steps):
        return self.env.decrement_starting_point(nr_steps)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

class DemoEnv(GymWrapper):

    def __init__(self, env, demo_file_name):
        super(GymWrapper, self).__init__(env)

        with open(os.path.join("demo", demo_file_name), "rb") as file:
            dat = pickle.load(file)

        self.checkpoints = dat['checkpoints']
        self.checkpoint_action_nr = dat['checkpoint_action_nr']
        self.actions = dat['actions']
        rewards = dat['rewards']
        self.returns = np.cumsum(rewards)
        self.starting_point = 1800
        self.reset_steps_ignored = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if reward != 0:
            done = True

        self.score += reward
        if (self.score > 0) and (self.score >= self.returns[self.action_nr]):
            self.decrement_starting_point(25)

        self.action_nr += 1
        info['starting_point'] = self.starting_point
        return observation, reward, done, info

    def decrement_starting_point(self, nr_steps):
        if self.starting_point > 0:
            self.starting_point = int(np.maximum(self.starting_point - nr_steps, 0))

    def reset(self):
        observation = self.env.reset()
        self.action_nr = 0
        start_checkpoint = None
        for nr, checkpoint in zip(self.checkpoint_action_nr[::-1], self.checkpoints[::-1]):
            if nr <= (self.starting_point - self.reset_steps_ignored):
                self.action_nr = nr
                start_checkpoint = checkpoint
                break

        self.score = self.returns[self.action_nr]
        if self.action_nr > 0:
            self.env.unwrapped.restore_state(start_checkpoint)
        return observation

class MaxMergeSkipEnv(GymWrapper):
    def __init__(self, env, skip=4):
        GymWrapper.__init__(self, env)
        self._observation_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        combined_info = {}

        for _ in range(self._skip):
            observation, reward, done, info = self.env.step(action)
            self._observation_buffer.append(observation)
            total_reward += reward
            combined_info.update(info)
            if done:
                break

        max_frame = np.max(self._observation_buffer, axis=0)
        return max_frame, total_reward, done, combined_info

    def reset(self):
        self._observation_buffer.clear()
        observation = self.env.reset()
        self._observation_buffer.append(observation)
        return observation

class BlackWhiteEnv(GymWrapper):
    def __init__(self, env):
        GymWrapper.__init__(self, env)

    def preprocess(self, image):
        image = image[35:195]
        image = image[::2, ::2, 0]
        player = (image == 92) * 1
        ball = (image == 236) * 1
        enemy = (image == 213) * 1
        return player + ball + enemy

    def reset(self):
        return self.preprocess(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.preprocess(observation), reward, done, info