from my_env import EvidenceEnv
import numpy as np

class RandomAgent(object):

    def __init__(self, env):
        self.env = env

    def act(self, observation):
        return np.random.choice(self.env.n_action)

    def train(self, a, old_obs, r, new_obs):
        pass

def main():
    #number of iterations
    n_iter = 1000
    #environment specs
    env = EvidenceEnv(n=2, p=0.95)
    #define agent
    agent = RandomAgent(env)
    #reset environment and agent
    obs = env.reset()
    reward = None
    done = False
    R = []
    for step in range(n_iter):
        env.render
        action = agent.act(obs)
        _obs, reward, done, _ = env.step(action)
        #no training involved for random agent
        agent.train(action, obs, reward, _obs)
        obs = _obs
        R.append(reward)

if __name__ == '__main__':
    main()
