import numpy as np


class MyEnv(object):
    def __init__(self, gym_env, history_length):
        self.env = gym_env
        self.history_length = history_length
        self.s_t = None
        self.pong_actions = [1, 2, 3]

    def reset(self):
        x_t = self.env.reset()
        x_t = self.pre_process(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        self.s_t = s_t
        return s_t

    def step(self, action):
        x_t1, r, done, info = self.env.step(self.pong_actions[action])
        x_t1 = self.pre_process(x_t1)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, self.s_t[:, :, :3], axis=2)
        self.s_t = s_t1
        return s_t1, r, done, info

    def pre_process(self, x):
        x = x[35:195]
        x = x[::2, ::2, 0]
        x[x == 144] = 0
        x[x == 109] = 0
        x[x != 0] = 1
        return x