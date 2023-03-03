import numpy as np
import numpy.random as npr

from daisy.daisy_world_rl import RLDaisyWorld

from daisy.nn.functional import glorot


class MLP():
    def __init__(self, **kwargs):

        self.in_dim = 45
        self.out_dim = 9
        self.h_dim = [16, 64]

        # relu activation
        self.act = lambda x: x * (x>0.0)

        self.initialize_parameters()

    def initialize_parameters(self):
        self.layers = []

        self.layers.append(glorot((self.in_dim, self.h_dim[0])))

        for ii in range(1, len(self.h_dim)):
            self.layers.append(\
                    glorot((self.h_dim[ii-1], self.h_dim[ii])))

        self.layers.append(glorot((self.h_dim[-1], self.out_dim)))


    def forward(self, x):

        for layer in self.layers[:-1]:
            x = self.act(np.matmul(x, layer))

        x = np.matmul(x, self.layers[-1])

        return x

    def get_action(self, obs):

        # obs has shape batch x n_agents x channels x dim x dim

        x = obs.reshape(*obs.shape[:-3], self.in_dim)

        raw_logits = self.forward(x) 

        action = np.argmax(raw_logits, axis=-1, keepdims=True)

        return action

    def __call__(self, obs):

        return self.get_action(obs)

    def get_parameters(self):
        parameters = np.array([])

        for layer in self.layers:
            parameters = np.append(parameters, layer.ravel())

        return parameters

    def set_parameters(self, parameters):

        param_start = 0
        param_stop = self.in_dim * self.h_dim[0]
        param_shapes = [self.in_dim]
        param_shapes.extend(self.h_dim)
        param_shapes.append(self.out_dim)


        for ii, layer in enumerate(self.layers):
        
            param_stop = param_start \
                    + param_shapes[ii] * param_shapes[ii+1]

            self.layers[ii] = parameters[param_start:param_stop]\
                    .reshape(param_shapes[ii], param_shapes[ii+1])

            param_start = param_stop

    def reset(self):
        pass
            
        
if __name__ == "__main__":

    env = RLDaisyWorld()

    agent = MLP()

    obs = env.reset()
    action = agent(obs)
    action = agent.get_action(obs)

    parameters = agent.get_parameters()
    agent_b = MLP()
    agent_b.set_parameters(parameters)
    parameters_b = agent_b.get_parameters()

    print("should be 0.0: ", np.abs(parameters - parameters_b).sum())

    mlp_sum = 0.0
    for ii in range(env.ramp_period*3):
        action = agent(obs)
        obs, r, d, i = env.step(action)
        alive = env.grid[:,1:3,:,:].sum()

        mlp_sum += r.item()

    print(f"mlp alive: {alive:.3f}")
    print(f"mlp reward sum: {mlp_sum:.3f}")
