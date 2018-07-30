from gym.envs.classic_control.rendering import SimpleImageViewer
import gym
import time
import pickle
import os

def test():
    with open(os.path.join("demo", "Pong.demo"), "rb") as f:
        dat = pickle.load(f)
    viewer = SimpleImageViewer()
    env = gym.make('PongNoFrameskip-v4')
    checkpoint = dat['checkpoints'][18]
    checkpoint_action_nr = dat['checkpoint_action_nr'][18]
    env.reset()
    env.unwrapped.restore_state(checkpoint)

    t = 0
    while True:
        print("t ", t)
        action = dat['actions'][checkpoint_action_nr+t]
        observation, reward, done, _ = env.step(action)
        viewer.imshow(observation)
        if reward != 0:
            print("*** reset ***")
            env.reset()
            break
        time.sleep(0.5)
        t += 1

test()