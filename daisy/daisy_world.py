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

        # Stefan-Boltzmann constant
        self.sigma = 5.67*10**-8
        # positve constant for calculating temp
        self.q = .5
        # death rate for daisies (constant)
        self.gamma = 0.1
        # stellar luminosity R[0.,2.]
        self.L = 1.0
        # flux constant
        self.S = 1.0

        self.albedo_bare = 0.3
        self.albedo_light = 0.75
        self.albedo_dark = 0.1
        # optimal temperature for plant growth (Kelvin)
        self.temp_optimal = 295.5
        

        self.dt = 1.0
        self.initialize_neighborhood()

    def initialize_neighborhood(self):

        self.kernel = torch.ones(self.ch,1,3,3)
        self.kernel[:,:,1,1] = 0.0
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
    
    def calculate_growth_rate(self, temp):

        beta = 1 - 0.003265*(self.temp_optimal - temp)**2
        return beta

    def calculate_growth(self, beta, grid, neighborhood):
        """
        """
        growth = torch.zeros_like(grid[:,1:3])

        # light daisies 
        i_l = grid[:, 1, ...] 
        n_l = neighborhood[:, 1, ...] 
        # dark daisies
        i_d = grid[:, 1, ...]
        n_d = neighborhood[:, 1, ...]

        dl_dt = n_l*((1-i_l-i_d)*beta - self.gamma)
        dd_dt = n_d*((1-i_l-i_d)*beta - self.gamma)

        # growth occurs in unoccupied or same-occupied cells
        growth[:,0,...] = (dl_dt > dd_dt) * (i_d <= 0.0)
        growth[:,1,...] = (dl_dt < dd_dt) * (i_l <= 0.0)

        self.growth = growth
        return growth

    def calculate_albedo(self, grid):

        # light daisies 
        i_l = grid[:, 1, ...] 
        # dark daisies
        i_d = grid[:, 1, ...]

        albedo = self.albedo_bare * (1 - (1.0 - i_d - i_l)) \
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

    def forward(self, grid):

        neighborhood = self.neighborhood_conv(grid)

        update = torch.zeros_like(grid)

        # daisy channels (light and dark)
        beta = self.calculate_growth_rate(\
                grid[:,self.ch-1:self.ch,...])
        update[:,1:3,:,:] = self.calculate_growth(\
                beta, grid, neighborhood)

        new_grid = grid + self.dt * update

        # daisy channels are updated, but temperature \
        # and albedo are a state based on albedo
        # albedo
        new_grid[:,0,:,:] = self.calculate_albedo(grid) 
        # temperature
        new_grid[:,3,:,:] = self.calculate_temperature(\
                grid, neighborhood) 

        return new_grid

    def step(self, action):
        pass

    def __call__(self, grid): 

        pass


if __name__ == "__main__":

    env = RLDaisyWorld()

    a = torch.rand(1,env.ch,16,16)

    b = env.forward(a)

    print(a.shape, b.shape)

