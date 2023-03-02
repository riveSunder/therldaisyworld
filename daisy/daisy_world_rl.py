
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib

def pad_to_2d(kernel, dims): 
                                
    mid_x = dims[-2] // 2                                                        
    mid_y = dims[-1] // 2                                                        
    mid_k_x = kernel.shape[-2] // 2                                              
    mid_k_y = kernel.shape[-1] // 2                                              
                                                                                
    start_x = mid_x - mid_k_x                                                   
    start_y = mid_y - mid_k_y                                                   
                                                                                
    padded = np.zeros(dims)                                                     
    padded[..., mid_x-mid_k_x:mid_x-mid_k_x + kernel.shape[-2],
            mid_y-mid_k_y:mid_y-mid_k_y + kernel.shape[-1]] = kernel             
                                                                                
    return padded                         
                                                        
def ft_convolve(grid, kernel):                                                  
                                             
    grid = np.array(grid)
    kernel = np.array(kernel)

    grid2 = grid           
    
    if np.shape(kernel) != np.shape(grid2):                                     
        padded_kernel = pad_to_2d(kernel, grid2.shape)                          
    else:                                                                       
        padded_kernel = kernel                                                  
                                                                                
#    convolved = (np.fft.ifftshift(np.real(np.fft.ifft2(\
#            np.fft.fft2(np.fft.fftshift(grid2)) \
#            * np.fft.fft2(np.fft.fftshift(padded_kernel))))))
    convolved = np.fft.ifftshift(\
                    np.real(np.fft.ifft2(\
                        np.fft.fft2(np.fft.fftshift(grid2, axes=(-2,-1)), axes=(-2,-1)) \
            * np.fft.fft2(np.fft.fftshift(padded_kernel, axes=(-2,-1)), axes=(-2,-1)), axes=(-2,-1)))\
                , axes=(-2,-1))
                                                                                
    return convolved 
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
        # starvation/food depletion for agents
        self.agent_gamma = 0.05
        self.q = 0.2 * self.S / self.sigma
        self.Toptim = 295.5
        self.dt = 0.01
        self.ddL = 0.
        
        # stellar luminosity R[0.,2.]
        self.max_L = 1.2
        self.min_L = 0.65
        self.initial_L = self.min_L
        self.ramp_period = 256 

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

        self.n_agents = 1

        self.initialize_neighborhood()
        self.initialize_agents()
        self.reset()

    def initialize_agents(self):

        self.agent_indices = torch.randint(self.dim, \
                size=(self.batch_size, self.n_agents, 2))

        # the "bellies" of the agents
        self.agent_states = torch.ones(self.batch_size, self.n_agents, 1)

    def update_agents(self, action):

        # TODO: agents aren't allowed to occupy the same cells in a grid
        # (required for multiagent env mode)
        self.agent_states = torch.clamp(self.agent_states - self.agent_gamma, 0.,1.)

        for bb in range(action.shape[0]):
            for nn in range(action.shape[1]):
                
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

                self.agent_indices[bb,nn,:] = self.agent_indices % self.dim

                if action[bb,nn,0] > 4:
                    # actions 4 through 8 indication grazing movement
                    xx, yy = self.agent_indices[bb,nn,0], self.agent_indices[bb,nn,1]
                    self.agent_states[bb,nn,0] += self.grid[bb,1:3,xx,yy].sum()
                    self.grid[bb,1:3,xx,yy] *= 0.0


    def get_obs(self, agent_indices=None):

        obs = torch.zeros(*agent_indices.shape[:2], self.ch, 3, 3)
        obs_grid = F.pad(self.grid, (1,1,1,1), mode="circular") 

        for bb in range(obs.shape[0]):
            for nn in range(obs.shape[1]):
                x_start = agent_indices[bb,nn,0]
                y_start = agent_indices[bb,nn,1]

                obs[bb,nn,:,:,:] = obs_grid[\
                        bb,:,x_start:x_start+3, y_start:y_start+3]

        return obs

    def initialize_neighborhood(self):

        ## Convolution for daisy density
        # central kernel weight is 0.67, adjacent weights = .37/8
        self.n_daisies = 2
        self.daisy_kernel = torch.ones(1,1,3,3) * np.exp(-1)
        self.daisy_kernel[:,:,1,1] = 1.0 
        self.daisy_kernel[:,:,0::2, 0::2] = np.exp(-2) 
        self.daisy_kernel /= self.daisy_kernel.sum() # self.ch * self.kernel / self.kernel.sum()
        kernel_dim = self.daisy_kernel.shape[-1]
        padding = 0 #(kernel_dim - 1) // 2

        self.daisy_conv = nn.Conv2d(1, 1,\
                kernel_dim, padding=padding,\
                padding_mode="reflect", \
                bias=False)

        for param in self.daisy_conv.named_parameters():
            param[1].requires_grad = False
            param[1][:] = self.daisy_kernel

        # local and adjacent albedo convs
        self.local_albedo_kernel = torch.zeros(1, 1, 3,3)
        self.local_albedo_kernel[:,:,1,1] = 1.0
        self.adjacent_albedo_kernel = torch.ones(1, 1, 3,3) \
                * 1/8. 
        self.adjacent_albedo_kernel[:,:,0,0] = 0.
        kernel_dim = self.local_albedo_kernel.shape[-1]
        padding = 0 #(kernel_dim - 1) // 2

        self.local_albedo_conv = nn.Conv2d(1, 1,
                kernel_dim, padding=padding,\
                padding_mode="reflect", \
                bias=False)

        self.adjacent_albedo_conv = nn.Conv2d(1, 1,
                kernel_dim, padding=padding,\
                padding_mode="reflect", \
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
        self.initialize_grid()

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

            local_albedo += albedo * groundcover[:,ii:ii+1,:,:]
            #* self.local_albedo_conv(groundcover[:,ii:ii+1,:,:]) 
#            temp_conv = self.adjacent_albedo_conv(\
#                    F.pad(groundcover[:,ii:ii+1,:,:], (self.dim,self.dim,self.dim,self.dim),\
#                    mode="circular"))[:,:,self.dim+1:1+2*self.dim,1+self.dim:1+self.dim*2]
#            adjacent_albedo += albedo * temp_conv #self.adjacent_albedo_conv(groundcover[:,ii:ii+1,:,:])
            adjacent_albedo += albedo * torch.tensor(ft_convolve(\
                    groundcover[:,ii:ii+1,:,:], self.adjacent_albedo_kernel))
    
        return local_albedo, adjacent_albedo

    def calculate_temperature(self, local_albedo, adjacent_albedo):

        # local albedo 
        Al = local_albedo
        # neighborhood albedo
        A = adjacent_albedo #.mean()
        
        # effective radiation temperature
        self.temp_effective = (\
                (self.S*self.L * (1-A))/self.sigma)**(1/4)

        dead_effective = (\
                (self.S * self.L * (1-self.albedo_bare))/self.sigma)**(1/4)
        self.dead_temp = dead_effective

        temp = (self.q*(A - Al) + self.temp_effective**4)**(1/4)
        self.temp = temp

        return temp

    def calculate_daisy_density(self, local_daisies):

        daisy_density = torch.zeros(1,2, self.dim, self.dim)

        for jj in range(self.n_daisies):
#            temp_conv = self.daisy_conv(F.pad(local_daisies[:,jj:jj+1,:,:],\
#                    (self.dim,self.dim,self.dim,self.dim), \
#                    mode="circular"))[:,:,1+self.dim:1+2*self.dim, 1+self.dim:1+2*self.dim]
#            daisy_density[:,jj:jj+1,:,:] = temp_conv#self.daisy_conv(local_daisies[:,jj:jj+1,:,:])
            daisy_density[:,jj:jj+1,:,:] = torch.tensor(\
                    ft_convolve(local_daisies[:,jj:jj+1,:,:], self.daisy_kernel))


        return daisy_density

    def forward(self, grid):

        local_albedo, adjacent_albedo = self.calculate_albedo(grid[:,:3,:,:])
        daisy_density = self.calculate_daisy_density(grid[:,1:3,:,:])

        temp = self.calculate_temperature(local_albedo, adjacent_albedo)
        beta = self.calculate_growth_rate(temp)
        growth = self.calculate_growth(beta, daisy_density)

        new_grid = 0. * grid
        grid[:,3,:,:] = temp
        new_grid[:,1:3, :,:] = torch.clamp(grid[:,1:3, :,:] + self.dt * growth, 0,1)
        new_grid[:,0, :,:] = self.p - new_grid[:,1, :,:] - new_grid[:,2,:,:] #.sum(dim=1)

        new_grid = torch.round(new_grid, decimals=3)

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

    def step(self, action=None):
        
        if action is None and self.n_agents:
            action = torch.zeros(self.batch_size, self.n_agents,1)

        if action is not None:
            self.update_agents(action)

        for ii in range(int(1. / self.dt)):
            self.grid = self.forward(self.grid) 

        obs = self.get_obs(self.agent_indices)
        reward = 1.0 * self.agent_states.detach()
        done = reward < 0.1

        done, info = 0, {}

        self.L = self.update_L(self.L)

        return obs, reward, done, info

    def __call__(self, grid): 

        pass


if __name__ == "__main__":

    env = RLDaisyWorld()

    a = env.grid

    b = env.forward(a)

    for ii in range(9):
        for jj in range(1):
            action = torch.tensor([[[ii]]]) #randint(9, size=(env.batch_size, env.n_agents, 1))
            
            print(env.grid[:,4,:,:])

            env.step(action)


    print(a.shape, b.shape)

