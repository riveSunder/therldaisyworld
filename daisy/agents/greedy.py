import numpy as np

from daisy.daisy_world_rl import RLDaisyWorld

class Greedy():

    def __init__(self, **kwargs):
        
        self.epsilon = kwargs["epsilon"] if "epsilon" in kwargs.keys() else 0.0
        self.greedy = kwargs["greedy"] if "greedy" in kwargs.keys() else True

        self.move_mask = np.array([[[1,3,5,7]]])

    def __call__(self, obs):

        daisy_sum = obs[...,1,:,:] + obs[...,2,:,:]

        daisy_sum = daisy_sum.reshape(*daisy_sum.shape[0:-2], -1)

        masked_daisies = daisy_sum[:,:,self.move_mask] 


        if np.random.rand() > self.epsilon:

            if self.greedy:
                action_arg = np.argmax(masked_daisies, axis=-1)
            else:
                action_arg = np.argmin(masked_daisies, axis=-1)

            action = 4 + action_arg
        else:
            action = np.random.randint(9, size=(*obs.shape[0:2],1,1))

        action = action.reshape(*obs.shape[0:2], -1)

        return action


if __name__ == "__main__":

    env = RLDaisyWorld()
    env.max_L = 1.5
    env.min_L = 1.4
    env.ramp_period = 100
    env.n_agents = 64
    obs = env.reset()
    np.random.seed(42)

    agent = Greedy()

    greedy_sum = 0.0
    for ii in range(env.ramp_period*3):
        action = agent(obs)
        obs, r, d, i = env.step(action)
        alive = env.grid[:,1:3,:,:].sum()

        greedy_sum += r.sum().item()

    print(f"greedy alive: {alive:.3f}")
    obs = env.reset()

    agent = Greedy()
    agent.epsilon = 1.0

    random_sum = 0.0
    for ii in range(env.ramp_period*3):
        action = agent(obs)
        obs, r, d, i = env.step(action*0)
        alive = env.grid[:,1:3,:,:].sum()

        random_sum += r.sum().item()

    print(f"random alive: {alive:.3f}")
    obs = env.reset()

    agent = Greedy(greedy=False)

    antigreedy_sum = 0.0
    for ii in range(env.ramp_period*3):
        action = agent(obs)
        obs, r, d, i = env.step(action)
        alive = env.grid[:,1:3,:,:].sum()

        random_sum += r.sum().item()

    print(f"anti-greedy alive: {alive:.3f}")
    print(f"reward in {ii} steps greedy: {greedy_sum:.3f}, "\
            f"anti-greedy: {antigreedy_sum:.3f}, "\
            f"random: {random_sum:.3f}")



