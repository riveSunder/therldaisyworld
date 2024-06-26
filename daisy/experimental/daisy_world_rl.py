import os

import json

import numpy as np
import numpy.random as npr
import time

from daisy.nn.functional import ft_convolve,\
        pad_to_2d

class RLDaisyWorld():

    def __init__(self):

        # channels
        self.ch = 5
        # batch_size 
        self.batch_size = 4
        # size of the toroidal daisyworld
        self.dim = 8

        # model parameters
        self.p = 1.00 #1.0
        self.g = 0.003265
        self.S = 1000.0
        # Stefan-Boltzmann constant
        self.sigma = 5.67e-8
        self.gamma = 0.05
        # starvation/food depletion for agents
        self.agent_gamma = 0.05
        self.q = 0.2 * self.S / self.sigma
        self.Toptim = 295.5
        self.dt = 0.1
        self.ddL = 0.
        
        # stellar luminosity R[0.,2.]
        self.max_L = 1.2
        self.min_L = 0.7
        self.initial_L = self.min_L
        self.ramp_period = 256 

        # seasonal parameters
        self.use_seasons = True
        self.use_inclination = True
        self.amplitude_seasonal = 0.05
        self.period_seasonal = self.ramp_period // 4
        self.max_tilt = 0.1

        self.albedo_bare = 0.5
        self.albedo_light = 0.75
        self.albedo_dark = 0.25
        # optimal temperature for plant growth (Kelvin)
        self.temp_optimal = 295.5
        
        # proportion of daisies per cell
        self.initial_al = 0.2
        self.initial_ad = 0.2
        # proportion of cells with daisies
        self.light_proportion = 0.33
        self.dark_proportion = 0.33

        self.n_agents = 3

        self.initialize_neighborhood()
        self.initialize_agents()
        self.reset()


    def make_config(self):

        config = {}

        config["max_L"] = self.max_L
        config["min_L"] = self.min_L
        config["initial_L"] = self.initial_L
        config["ramp_period"] = self.ramp_period
        config["dL"] = self.dL
        config["p"] = self.p
        config["g"] = self.g
        config["S"] = self.S
        config["sigma"] = self.sigma
        config["gamma"] = self.gamma
        config["albedo_bare"] = self.albedo_bare
        config["albedo_light"] = self.albedo_light
        config["albedo_dark"] = self.albedo_dark
        config["temp_optimal"] = self.temp_optimal
        config["light_proportion"] = self.light_proportion
        config["dark_proportion"] = self.dark_proportion
        config["initial_al"] = self.initial_al
        config["initial_ad"] = self.initial_ad
        config["n_agents"] = self.n_agents
        config["agent_gamma"] = self.agent_gamma

        return config

    def save_config(self, filepath=None):

        if filepath is None:
            filepath = os.path.join("results", "default_model_config.json")

        config = self.make_config()

        with open(filepath, "w") as f:
            json.dump(config, f)

    def _apply_config(self, config):

        self.max_L = config["max_L"]
        self.min_L = config["min_L"]
        self.initial_L = config["initial_L"]
        self.ramp_period = config["ramp_period"]
        self.dL = config["dL"]
        self.p = config["p"]
        self.g = config["g"]
        self.S = config["S"]
        self.sigma = config["sigma"]
        self.gamma = config["gamma"]
        self.albedo_bare = config["albedo_bare"]
        self.albedo_light = config["albedo_light"]
        self.albedo_dark = config["albedo_dark"]
        self.temp_optimal = config["temp_optimal"]
        self.light_proportion = config["light_proportion"]
        self.dark_proportion = config["dark_proportion"]
        self.initial_al = config["initial_al"]
        self.initial_ad = config["initial_ad"]
        self.n_agents = config["n_agents"]
        self.agent_gamma = config["agent_gamma"]

    def load_config(self, filepath=None):

        if filepath is None:
            filepath = os.path.join("results", "default_model_config.json")

        with open(filepath, "r") as f:
            config = json.load(f)

        return config

    def restore_config(self, filepath=None):

        if filepath is None:
            filepath = os.path.join("results", "default_model_config.json")

        config = self.load_config(filepath)

        self._apply_config(config)

    def initialize_agents(self):

        self.agent_indices = np.random.randint(self.dim, \
                size=(self.batch_size, self.n_agents, 2))

        # the "bellies" of the agents
        self.agent_states = np.ones((self.batch_size, self.n_agents, 1))

    def update_agents(self, action):

        # TODO: agents aren't allowed to occupy the same cells in a grid
        # (required for multiagent env mode)
        self.agent_states = np.clip(self.agent_states - self.agent_gamma, 0.,1.)

        for bb in range(action.shape[0]):
            for nn in range(action.shape[1]):
                # dead agents don't move
                if self.agent_states[bb,nn] > 0.0:
                    
                    if action[bb,nn,0] == 8:
                        pass
                        # no eating or movement
                    elif action[bb,nn,0] % 4 == 0:
                        self.agent_indices[bb,nn,1] -= 1
                    elif action[bb,nn,0] % 4 == 1:
                        self.agent_indices[bb,nn,0] -= 1
                    elif action[bb,nn,0] % 4 == 2:
                        self.agent_indices[bb,nn,1] += 1
                    elif action[bb,nn,0] % 4 == 3:
                        self.agent_indices[bb,nn,0] += 1

                    self.agent_indices = self.agent_indices % self.dim

                    if action[bb,nn,0] > 4:
                        # actions 4 through 8 indication grazing movement
                        xx, yy = self.agent_indices[bb,nn,0], self.agent_indices[bb,nn,1]
                        self.agent_states[bb,nn,0] += self.grid[bb,1:3,xx,yy].sum()
                        self.grid[bb,1:3,xx,yy] *= 0.0


    def get_obs(self, agent_indices=None):

        obs = np.zeros((*agent_indices.shape[:2], self.ch, 3, 3))
        pad_dims = (*self.grid.shape[:-2], self.grid.shape[-2] + 2, self.grid.shape[-1] + 2)
        obs_grid = pad_to_2d(self.grid, dims=pad_dims, mode="circular") 


        for bb in range(obs.shape[0]):
            for nn in range(obs.shape[1]):
                x_start = agent_indices[bb,nn,0]
                y_start = agent_indices[bb,nn,1]

                obs[bb,nn,:,:,:] = obs_grid[\
                        bb,:,x_start:x_start+3, y_start:y_start+3]

        return obs

    def initialize_neighborhood(self):

        ## Convolution for daisy density
        # kernel based on Gaussian
        self.n_daisies = 2
        self.daisy_kernel = np.ones((1,1,3,3)) * np.exp(-1)
        self.daisy_kernel[:,:,1,1] = 1.0 
        self.daisy_kernel[:,:,0::2, 0::2] = np.exp(-2) 
        self.daisy_kernel /= self.daisy_kernel.sum() 
        kernel_dim = self.daisy_kernel.shape[-1]
        padding = 0 #(kernel_dim - 1) // 2

        # local and adjacent albedo convs
        self.local_albedo_kernel = np.zeros((1, 1, 3,3))
        self.local_albedo_kernel[:,:,1,1] = 1.0
        self.adjacent_albedo_kernel = np.ones((1, 1, 3,3)) / 8. 
        self.adjacent_albedo_kernel[:,:,0,0] = 0.
        kernel_dim = self.local_albedo_kernel.shape[-1]
        padding = 0 

    def initialize_grid(self):

        dark_probability = np.random.rand(\
                self.batch_size,\
                2,\
                self.dim,\
                self.dim)

        light_probability = np.random.rand(\
                self.batch_size,\
                2,\
                self.dim,\
                self.dim)

        dark_daisies = 1.0 * (dark_probability[:,0,:,:] < self.dark_proportion) \
                * self.initial_ad * dark_probability[:,1,:,:] 
        light_daisies = 1.0 * (light_probability[:,0,:,:] < self.light_proportion) \
                * self.initial_al * light_probability[:,1,:,:] 

        grid =  np.zeros((\
                self.batch_size,\
                self.ch,\
                self.dim,\
                self.dim))

        grid[:,0,...] = self.p - light_daisies - dark_daisies
        grid[:,1,...] = light_daisies
        grid[:,2,...] = dark_daisies

        local_albedo, adjacent_albedo = self.calculate_albedo(grid[:,:3,:,:])
        daisy_density = self.calculate_daisy_density(grid[:,1:3,:,:])

        temp = self.calculate_temperature(local_albedo, adjacent_albedo)
        beta = self.calculate_growth_rate(temp)
        growth = self.calculate_growth(beta, daisy_density)

        grid[:,3:4,:,:] = temp
        self.grid = grid


    def reset(self):

        self.L = self.min_L
        self.dL =  (self.max_L - self.min_L) / self.ramp_period

        self.step_count = 0
        self.update_inclination()
        self.initialize_grid()
        self.initialize_agents()

        obs = self.get_obs(self.agent_indices)

        return obs

    def calculate_growth_rate(self, temp):
        beta = 1 - self.g*(self.temp_optimal - temp)**2 
        self.beta = beta
        return beta

    def calculate_growth(self, beta, daisy_density):
        """
        """

        # light daisies 
        a_l = daisy_density[:, 0, :, :] 
        # dark daisies
        a_d = daisy_density[:, 1, :, :] 

        # bare ground available for growth 
        a_b = self.p - a_l - a_d 

        dl_dt = a_l*(a_b * beta.squeeze() - self.gamma)
        dd_dt = a_d*(a_b * beta.squeeze() - self.gamma)

        growth = np.zeros_like((daisy_density))
        growth[:,0,...] = dl_dt 
        growth[:,1,...] = dd_dt

        self.growth = growth

        return growth

    def calculate_albedo(self, groundcover):

        # groundcover has 3 channels (bare, light daisies, dark daisies)

        groundcover[:,0,...] = self.p - groundcover[:,1,:,:] - groundcover[:,2,:,:] 
        local_albedo = np.zeros(\
                (self.batch_size,1,self.dim, self.dim) )
        adjacent_albedo = np.zeros(\
                (self.batch_size,1,self.dim, self.dim))

        for ii, albedo in enumerate([\
                self.albedo_bare, self.albedo_light, self.albedo_dark]):

            local_albedo += albedo * groundcover[:,ii:ii+1,:,:]
            adjacent_albedo += albedo * ft_convolve(\
                    groundcover[:,ii:ii+1,:,:], self.adjacent_albedo_kernel)
    
        return local_albedo, adjacent_albedo

    def calculate_temperature(self, local_albedo, adjacent_albedo):

        # local albedo 
        Al = local_albedo
        # neighborhood albedo
        A = adjacent_albedo #.mean()
        
        # effective radiation temperature
        self.temp_effective = (\
                (self.S*self.L * self.inclination * (1-A))/self.sigma)**(1/4)

        dead_effective = (\
                (self.S * self.L * self.inclination * (1-self.albedo_bare))/self.sigma)**(1/4)
        self.dead_temp = dead_effective

        temp = (self.q*(A - Al) + self.temp_effective**4)**(1/4)
        self.temp = temp

        return temp

    def calculate_daisy_density(self, local_daisies):

        daisy_density = np.zeros((self.batch_size,2, self.dim, self.dim))

        for jj in range(self.n_daisies):
            daisy_density[:,jj:jj+1,:,:] = ft_convolve(local_daisies[:,jj:jj+1,:,:], \
                    self.daisy_kernel)


        return daisy_density

    def forward(self, grid):

        local_albedo, adjacent_albedo = self.calculate_albedo(grid[:,:3,:,:])
        daisy_density = self.calculate_daisy_density(grid[:,1:3,:,:])

        temp = self.calculate_temperature(local_albedo, adjacent_albedo)
        beta = self.calculate_growth_rate(temp)
        growth = self.calculate_growth(beta, daisy_density)

        new_grid = 0. * grid
        grid[:,3:4,:,:] = temp
        new_grid[:,1:3, :,:] = np.clip(grid[:,1:3, :,:] + self.dt * growth, 0,1)
        new_grid[:,0, :,:] = self.p - new_grid[:,1, :,:] - new_grid[:,2,:,:] #.sum(dim=1)

        new_grid = np.round(new_grid, decimals=3)

        if self.n_agents:
            for bb in range(self.batch_size):
                for nn in range(self.n_agents):
                    
                    xx, yy = self.agent_indices[bb,nn,0], self.agent_indices[bb,nn,1]
                    new_grid[bb,4,xx,yy] = self.agent_states[bb,nn]

        return new_grid

    def update_L(self, L):
    
        self.step_count += 1
        if self.step_count % self.ramp_period == 0:
            self.dL *= -1
            self.min_L -= self.ddL
            self.max_L += self.ddL

        L += self.dL

        return max([min([L, self.max_L]), self.min_L])

    def update_inclination(self):

        if self.use_inclination:

            offset = self.max_tilt * np.sin(self.step_count * 2 * np.pi / self.period_seasonal)

            inclination_x = np.arange(-np.pi + offset, \
                    np.pi * (1 + 2/(self.dim-1))+ offset, \
                    2*np.pi / (self.dim-1))

            yy, xx = np.meshgrid(inclination_x, inclination_x)

            xx =  np.cos(xx)
            self.inclination = 1.0 + self.amplitude_seasonal * xx 

            # inclination modifier is adjusted to have mean luminosity 1.0
            #self.inclination /= self.inclination.mean()
        else:

            self.inclination = np.ones((1,1,self.dim, self.dim))

    def step(self, action=None):
        
        if action is None and self.n_agents:
            action = np.zeros((self.batch_size, self.n_agents,1))

        if action is not None:
            self.update_agents(action)

        for ii in range(int(1. / self.dt)):
            self.grid = self.forward(self.grid) 

        obs = self.get_obs(self.agent_indices)
        if self.n_agents:
            reward = 1.0 * self.agent_states
        else:
            reward = self.grid[:,1:3,:,:].sum(axis=(-2,-1)) > 0

        reward = reward * (reward > 0)
        done = reward < 0.1

        info = {}
        self.L = self.update_L(self.L)

        if self.use_seasons:
            self.update_inclination()

        return obs, reward, done, info

    def __call__(self, grid): 

        pass


if __name__ == "__main__":

    env = RLDaisyWorld()

    a = env.grid

    b = env.forward(a)

    for ii in range(9):
        for jj in range(1):
            action = np.array([[[ii]]]) #randint(9, size=(env.batch_size, env.n_agents, 1))
            
            print(env.grid[:,4,:,:])

            env.step(action)


    print(a.shape, b.shape)

