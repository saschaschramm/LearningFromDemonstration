import numpy as np
from common.stats_recorder import StatsRecorder
from common.env_wrapper import action_with_index

def discount(rewards, dones, discount_rate):
    discounted = []
    total_return = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        if done:
            total_return = reward
        else:
            total_return = reward + discount_rate * total_return
        discounted.append(total_return)
    return np.asarray(discounted[::-1])

class Runner():

    def __init__(self,
                 env,
                 model,
                 batch_size,
                 timesteps,
                 discount_rate,
                 summary_frequency,
                 performance_num_episodes,
                 summary_log_dir):
        self.env = env
        self.model = model
        self.timesteps = timesteps

        self.discount_rate = discount_rate
        self.observation = env.reset()
        self.batch_size = batch_size
        self.stats_recorder = StatsRecorder(summary_frequency=summary_frequency,
                                            performance_num_episodes=performance_num_episodes,
                                            summary_log_dir=summary_log_dir,
                                            save=True)

    def run(self):
        batch_observations = []
        batch_rewards = []
        batch_actions = []
        batch_dones = []

        for t in range(self.timesteps+1):
            action_index = self.model.predict_action([self.observation])[0]
            batch_observations.append(self.observation)

            action = action_with_index(action_index)
            self.observation, reward, done, info = self.env.step(action)

            if t % self.stats_recorder.summary_frequency == 0:
                print(info["starting_point"])

            self.stats_recorder.after_step(reward=reward, done=done, t=t)

            batch_rewards.append(reward)
            batch_actions.append(action_index)
            batch_dones.append(done)

            if len(batch_rewards) == self.batch_size:
                discounted_reward = discount(batch_rewards, batch_dones, self.discount_rate)

                self.model.train(batch_observations, discounted_reward, batch_actions)
                batch_observations = []
                batch_rewards = []
                batch_actions = []
                batch_dones = []

            if done:
                self.observation = self.env.reset()

            if t % self.stats_recorder.summary_frequency == 0:
                self.model.save(0)