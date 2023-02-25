from autograd import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib

class RLDaisyWorld():

    def __init__(self):

        # channels
        self.ch = 5
        # batch_size 
        self.batch_size = 1
        self.dark_proportion = 0.3
        self.light_proportion = 0.3
        self.dim = 32

        # Stefan-Boltzmann constant
        self.sigma = 5.67*10**-8
        # positve constant for calculating temp
        self.q = .1
        # death rate for daisies (constant)
        self.gamma = 0.05
        # stellar luminosity R[0.,2.]
        self.S = 510
        self.L = 0.9
        self.max_L = 1.2
        self.min_L = 0.8
        self.ramp_period = 1000
        self.dL = 4 * (self.max_L - self.min_L) / self.ramp_period
        # flux constant

        self.albedo_bare = 0.5
        self.albedo_light = 0.75
        self.albedo_dark = 0.25
        # optimal temperature for plant growth (Kelvin)
        self.temp_optimal = 295.5
        

        self.dt = 1.0
        self.initialize_neighborhood()
        self.initialize_grid()

    def initialize_neighborhood(self):

        self.kernel = torch.ones(self.ch,1,3,3)
        #self.kernel[:,:,1,1] = 0.0
        self.kernel = self.kernel / self.kernel.sum()
        dim = self.kernel.shape[-1]
        padding = (dim - 1) // 2

        self.neighborhood_conv = nn.Conv2d(self.ch, self.ch,\
                dim, groups=self.ch, padding=padding,\
                padding_mode="circular", \
                bias=False)

        for param in self.neighborhood_conv.named_parameters():
            param[1].requires_grad = False
            param[1][:] = self.kernel

    def initialize_grid(self):

        dark_probability = torch.rand(\
                self.batch_size,\
                1,\
                self.dim,\
                self.dim)

        light_probability = torch.rand(\
                self.batch_size,\
                1,\
                self.dim,\
                self.dim)

        dark_daisies = 1.0 * (dark_probability < self.dark_proportion) 
        light_daisies = 1.0 * (light_probability < self.light_proportion) 

        # daisies can't grow in the same place
        no_daisy_mask = light_daisies * dark_daisies
        dark_daisies[no_daisy_mask > 0] = 0
        light_daisies[no_daisy_mask > 0] = 0

        grid =  torch.zeros(\
                self.batch_size,\
                self.ch,\
                self.dim,\
                self.dim)

        grid[:,1,...] = light_daisies
        grid[:,2,...] = dark_daisies

        neighborhood = self.neighborhood_conv(grid)

        beta = self.calculate_growth_rate(\
                grid[:,3,...])
        not_update = self.calculate_growth(\
                beta, grid, neighborhood)
        # albedo
        grid[:,0,:,:] = self.calculate_albedo(grid) 
        neighborhood = self.neighborhood_conv(grid)
        # temperature
        grid[:,3,:,:] = self.calculate_temperature(\
                grid, neighborhood) 
        self.grid = grid


    def reset(self):

        self.step_count = 0
        self.initialize_grid()

    def calculate_growth_rate(self, temp):
        beta = 1 - 0.003265*(self.temp_optimal - temp)**2 #, 0,1.)
        self.beta = beta
        return beta

    def calculate_growth(self, beta, grid, neighborhood):
        """
        """
        growth = torch.zeros_like(grid[:,1:3,:,:])

        # light daisies 
        i_l = grid[:, 1, ...] 
        n_l = neighborhood[:, 1, ...] 
        # dark daisies
        i_d = grid[:, 2, ...]
        n_d = neighborhood[:, 2, ...]

        dl_dt = n_l*(1-i_d)*beta - (self.gamma)
        dd_dt = n_d*(1-i_l)*beta - (self.gamma)

        # growth occurs in unoccupied or same-occupied cells
        growth[:,0,...] = dl_dt #- dd_dt
        #* (dl_dt > dd_dt) #* (i_d <= 0.0)
        growth[:,1,...] = dd_dt
        #* (dl_dt < dd_dt) #* (i_l <= 0.0)

        self.growth = growth
        return growth

    def calculate_albedo(self, grid):

        # light daisies 
        i_l = grid[:, 1, ...] 
        # dark daisies
        i_d = grid[:, 2, ...]

        albedo = self.albedo_bare * (1 - i_d - i_l) \
                + self.albedo_light * i_l + self.albedo_dark * i_d

        self.albedo = torch.clamp(albedo, 0, 1.0)
        return torch.clamp(albedo, 0, 1.0)

    def calculate_temperature(self, grid, neighborhood):

        # local albedo 
        i_a = grid[:,0]
        # neighborhood albedo
        n_a = neighborhood[:,0]
        
        # effective radiation temperature
        self.temp_effective = (\
                (self.S*self.L * (1-n_a))/self.sigma)**(1/4)

        temp = (self.q*(n_a - i_a) + self.temp_effective**4)**(1/4)
        self.temp = temp

        return temp

    def assign_daisies(self, grid):

        # discretize presence of daisies according to 
        # daisy channels as probabilities. 

        new_grid = 0. * grid

        daisy_dark_prob = torch.rand_like(grid[:,1,:,:])
        daisy_light_prob = torch.rand_like(grid[:,2,:,:])

        new_grid[:,2,:,:] = 1.0 * (grid[:,2,:,:] > daisy_dark_prob)
        new_grid[:,1,:,:] = grid[:,1,:,:] > daisy_light_prob

        new_grid[:,2,:,:] -= new_grid[:,1,:,:] \
                * (daisy_dark_prob < daisy_light_prob)
        new_grid[:,1,:,:] -= new_grid[:,2,:,:] \
                * (daisy_dark_prob >= daisy_light_prob)

        return torch.clamp(new_grid, 0.0, 1.0)

    def forward(self, grid):

        neighborhood = self.neighborhood_conv(grid)

        update = torch.zeros_like(grid)

        # daisy channels (light and dark)
        beta = self.calculate_growth_rate(\
                grid[:,3,...])
        update[:,1:3,:,:] = self.calculate_growth(\
                beta, grid, neighborhood)

        self.update = update
        new_grid = torch.clamp(grid + self.dt * update, 0., 1.0)
        new_grid = self.assign_daisies(new_grid)

        # daisy channels are updated, but temperature \
        # and albedo are a state based on albedo
        # albedo
        new_grid[:,0,:,:] = self.calculate_albedo(grid) 
        # temperature
        new_grid[:,3,:,:] = self.calculate_temperature(\
                grid, neighborhood) 

        return new_grid

    def update_L(self, L):
    

        self.step_count += 1
        if self.step_count % self.ramp_period == 0:
            self.dL *= -1
            self.min_L -= 0.01
            self.max_L += 0.01

        L += self.dL

        return max([min([L, self.max_L]), self.min_L])

        

    def step(self, action=None):
        
        self.grid = self.forward(self.grid) 

        reward, obs, done, info = 0, self.grid[:,0:3,...], 0, {}

        self.L = self.update_L(self.L)

        return reward, obs, done, info

    def __call__(self, grid): 

        pass


if __name__ == "__main__":

    env = RLDaisyWorld()

    a = torch.rand(1,env.ch,16,16)

    b = env.forward(a)

    print(a.shape, b.shape)

