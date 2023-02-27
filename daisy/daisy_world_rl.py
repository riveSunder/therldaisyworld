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
        # size of the toroidal daisyworld
        self.dim = 8

        # model parameters
        self.p = 1.00 #1.0
        self.g = 0.003265
        self.S = 1000.0
        # Stefan-Boltzmann constant
        self.sigma = 5.67e-8
        self.gamma = 0.05
        self.q = 0.2 * self.S / self.sigma
        self.Toptim = 295.5
        self.dt = 0.01
        self.ddL = 0.
        
        # stellar luminosity R[0.,2.]
        self.max_L = 1.15
        self.min_L = 0.75
        self.initial_L = self.min_L
        self.ramp_period = 100 #* 1./self.dt)

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

        self.initialize_neighborhood()
        self.reset()

    def initialize_neighborhood(self):

        ## Convolution for daisy density
        # central kernel weight is 0.67, adjacent weights = .37/8
        self.n_daisies = 2
        self.daisy_kernel = torch.ones(1,1,3,3) * 0.0465
        self.daisy_kernel[:,:,1,1] = 0.67
        #self.kernel = self.ch * self.kernel / self.kernel.sum()
        kernel_dim = self.daisy_kernel.shape[-1]
        padding = (kernel_dim - 1) // 2

        self.daisy_conv = nn.Conv2d(1, 1,\
                kernel_dim, padding=padding,\
                padding_mode="circular", \
                bias=False)

        for param in self.daisy_conv.named_parameters():
            param[1].requires_grad = False
            param[1][:] = self.daisy_kernel

        # local and adjacent albedo convs
        self.local_albedo_kernel = torch.zeros(1, 1, 3,3)
        self.local_albedo_kernel[:,:,1,1] = 1.0
        self.adjacent_albedo_kernel = torch.ones(1, 1, 3,3) \
                * 1/9. 
        kernel_dim = self.local_albedo_kernel.shape[-1]
        padding = (kernel_dim - 1) // 2

        self.local_albedo_conv = nn.Conv2d(1, 1,
                kernel_dim, padding=padding,\
                padding_mode="circular", \
                bias=False)

        self.adjacent_albedo_conv = nn.Conv2d(1, 1,
                kernel_dim, padding=padding,\
                padding_mode="circular", \
                bias=False)

        for param in self.local_albedo_conv.named_parameters():
            param[1].requires_grad = False
            param[1][:] = self.local_albedo_kernel

        for param in self.adjacent_albedo_conv.named_parameters():
            param[1].requires_grad = False
            param[1][:] = self.adjacent_albedo_kernel

    def initialize_grid(self):

        dark_probability = torch.rand(\
                self.batch_size,\
                2,\
                self.dim,\
                self.dim)

        light_probability = torch.rand(\
                self.batch_size,\
                2,\
                self.dim,\
                self.dim)

        dark_daisies = 1.0 * (dark_probability[:,0,:,:] < self.dark_proportion) \
                * self.initial_ad * dark_probability[:,1,:,:] 
        light_daisies = 1.0 * (light_probability[:,0,:,:] < self.light_proportion) \
                * self.initial_al * light_probability[:,1,:,:] 

        grid =  torch.zeros(\
                self.batch_size,\
                self.ch,\
                self.dim,\
                self.dim)

        grid[:,0,...] = self.p - light_daisies - dark_daisies
        grid[:,1,...] = light_daisies
        grid[:,2,...] = dark_daisies

        local_albedo, adjacent_albedo = self.calculate_albedo(grid[:,:3,:,:])
        daisy_density = self.calculate_daisy_density(grid[:,1:3,:,:])

        temp = self.calculate_temperature(local_albedo, adjacent_albedo)
        beta = self.calculate_growth_rate(temp)
        growth = self.calculate_growth(beta, daisy_density)

        grid[:,3,:,:] = temp
        self.grid = grid

    def reset(self):

        self.L = self.min_L
        self.dL = 2 * (self.max_L - self.min_L) / self.ramp_period

        self.step_count = 0
        self.dL = 2 * (self.max_L - self.min_L) / self.ramp_period
        self.initialize_grid()

    def calculate_growth_rate(self, temp):
        beta = 1 - self.g*(self.temp_optimal - temp)**2 #, 0,1.)
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

        dl_dt = a_l*(a_b * beta - self.gamma)
        dd_dt = a_d*(a_b * beta - self.gamma)

        growth = torch.zeros_like(daisy_density)
        growth[:,0,...] = dl_dt 
        growth[:,1,...] = dd_dt

        self.growth = growth

        return growth

    def calculate_albedo(self, groundcover):

        # groundcover has 3 channels (bare, light daisies, dark daisies)

        groundcover[:,0,...] = self.p - groundcover[:,1,:,:] - groundcover[:,2,:,:] 
        local_albedo = torch.zeros(1,1,self.dim, self.dim) 
        adjacent_albedo = torch.zeros(1,1,self.dim, self.dim) 

        for ii, albedo in enumerate([\
                self.albedo_bare, self.albedo_light, self.albedo_dark]):

            local_albedo += albedo * self.local_albedo_conv(groundcover[:,ii:ii+1,:,:])
            adjacent_albedo += albedo * self.adjacent_albedo_conv(groundcover[:,ii:ii+1,:,:])
    
        return local_albedo, adjacent_albedo

    def calculate_temperature(self, local_albedo, adjacent_albedo):

        # local albedo 
        Al = local_albedo
        # neighborhood albedo
        A = adjacent_albedo
        
        # effective radiation temperature
        self.temp_effective = (\
                (self.S*self.L * (1-A))/self.sigma)**(1/4)

        temp = (self.q*(A - Al) + self.temp_effective**4)**(1/4)
        self.temp = temp

        return temp

    def calculate_daisy_density(self, local_daisies):

        daisy_density = torch.zeros(1,2, self.dim, self.dim)

        for jj in range(self.n_daisies):
            daisy_density[:,jj:jj+1,:,:] = self.daisy_conv(local_daisies[:,jj:jj+1,:,:])

        return daisy_density

    def forward(self, grid):

        local_albedo, adjacent_albedo = self.calculate_albedo(grid[:,:3,:,:])
        daisy_density = self.calculate_daisy_density(grid[:,1:3,:,:])

        temp = self.calculate_temperature(local_albedo, adjacent_albedo)
        beta = self.calculate_growth_rate(temp)
        growth = self.calculate_growth(beta, daisy_density)

        new_grid = 1. * grid
        grid[:,3,:,:] = temp
        new_grid[:,1:3, :,:] = torch.clamp(grid[:,1:3, :,:] + self.dt * growth, 0,1)
        new_grid[:,0, :,:] = self.p - new_grid[:,1, :,:] - new_grid[:,2,:,:] #.sum(dim=1)


        return new_grid

    def update_L(self, L):
    
        self.step_count += 1
        if self.step_count % self.ramp_period == 0:
            self.dL *= -1
            self.min_L -= self.ddL
            self.max_L += self.ddL

        L += self.dL

        return max([min([L, self.max_L]), self.min_L])

    def step(self, action=None):
        
        for ii in range(int(1. / self.dt)):
            self.grid = self.forward(self.grid) 

        reward, obs, done, info = 0, self.grid[:,0:3,...], 0, {}

        self.L = self.update_L(self.L)

        return reward, obs, done, info

    def __call__(self, grid): 

        pass


if __name__ == "__main__":

    env = RLDaisyWorld()

    a = env.grid

    b = env.forward(a)

    print(a.shape, b.shape)

