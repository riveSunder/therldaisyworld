import numpy as np
import matplotlib.pyplot as plt

def tensor_to_image(grid):
    if len(grid.shape) == 4:
        image = grid.transpose(0,2,3,1)[0]
        
    if len(grid.shape) == 3:
        image = grid.transpose(1,2,0)

    return image

def plot_grid(env):

    global subplot_0
    global subplot_1
    global subplot_2
    global subplot_2b
    global subplot_2c
    global subplot_3
    global subplot_4
    global subplot_5
    global subplot_6
    
    global subplot_2_axis
    global subplot_4_axis
    global subplot_2b_axis
    global pop_dark
    global pop_light
    global mean_temp
    global dead_temp
    global std_temp
    global luminosity
    
    pop_light = []
    pop_dark = []
    mean_temp = []
    dead_temp = []
    std_temp = []
    luminosity = []
            
    fig, ax = plt.subplots(3,2, figsize=(5.25,5.25), facecolor="white")
    ax2 = ax[1,0].twinx()
    
    beta = tensor_to_image(env.beta)
    temp = tensor_to_image(env.temp)
    albedo = tensor_to_image(env.grid[:,:3,:,:]) #tensor_to_image(env.albedo)
    growth = tensor_to_image(env.growth)
    
    mean_temp.append(temp.mean())
    dead_temp.append(env.dead_temp)
    std_temp.append(temp.std())
    pop_light.append(env.grid[:,1,:,:].mean())
    pop_dark.append(env.grid[:,2,:,:].mean())
    luminosity.append(env.L)
    
    grid_display = tensor_to_image(env.grid)
    
    # albedo
    subplot_0 = ax[0,0].imshow(albedo, cmap="gray", interpolation="nearest",
                              vmin=0, vmax=1.0)
    # temp
    subplot_1 = ax[0,1].imshow(temp, cmap="afmhot", interpolation="nearest",\
                              vmin=250, vmax=350)
    subplot_5 = ax[2,0].imshow(growth[:,:,0], cmap="magma", vmin=-0.1, vmax=0.3)
    subplot_6 = ax[2,1].imshow(growth[:,:,1], cmap="magma", vmin=-0.1, vmax=0.3)
    
    # mean temp
    upper = [elem1 + elem2 for elem1, elem2 in zip(mean_temp, std_temp)]
    lower = [elem1 - elem2 for elem1, elem2 in zip(mean_temp, std_temp)]
    x = [count for count in range(len(upper))]
    
    subplot_2, = ax[1,0].plot(x, mean_temp, alpha=0.5, label="mean_temp")
    subplot_2c, = ax[1,0].plot(x, dead_temp, "-.", alpha=0.5, label="lifeless temp")
    subplot_2b, = ax2.plot(luminosity, "--", label="stellar luminosity")
    subplot_2b_axis = ax2.axis([0, len(upper), 0.5, 1.5])
    
    # habitable range
    t_range = np.sqrt(1 / 0.003265)
    
    habitable_x = [0, env.ramp_period*20]
    lower = [env.temp_optimal - t_range, env.temp_optimal - t_range]
    upper = [env.temp_optimal + t_range, env.temp_optimal + t_range]
    ax[1,0].fill_between(habitable_x, lower, upper, alpha=0.1)
    
    subplot_2_axis = ax[1,0].axis([0, len(upper), 278, 350])
    
    # population
    subplot_3, = ax[1,1].plot(x, pop_light, color=[0.7,0.7, 0.7], label="light population")
    subplot_4, = ax[1,1].plot(x, pop_dark, color=[0.2,0.2, 0.2], label="dark population")
    subplot_4_axis = ax[1,1].axis([0, len(x), 0, 1.0])
    
    for xx in range(1):
        for yy in range(2):
            ax[xx,yy].set_yticklabels('')
            ax[xx,yy].set_xticklabels('')

    plt.tight_layout()

    return fig, ax

def get_update_fig_old(env):

    def update_fig(ii):

        global subplot_0
        global subplot_1
        global subplot_2
        global subplot_3
        global subplot_4
        global subplot_5
        global subplot_6
        global subplot_2b
        global subplot_2c
        
        global subplot_2_axis
        global subplot_4_axis
        global subplot_2b_axis
        global pop_dark
        global pop_light
        global mean_temp
        global dead_temp
        global std_temp
        global env #daisy_world
        #global grid
        global ax
        global ax2
        global luminosity 
        global obs
        
        
        
        beta = tensor_to_image(env.beta)
        temp = tensor_to_image(env.temp)
        albedo = np.clip(\
                tensor_to_image(env.grid[:,:3,:,:]),0,1.) #tensor_to_image(env.albedo)
        growth = tensor_to_image(env.growth)
        
        mean_temp.append(temp.mean())
        dead_temp.append(env.dead_temp)
        std_temp.append(temp.std())
        pop_light.append(env.grid[:,1,:,:].mean())
        pop_dark.append(env.grid[:,2,:,:].mean())
        luminosity.append(env.L)
            
        subplot_0.set_array(albedo)
        subplot_1.set_array(temp)
        subplot_5.set_array(growth[:,:,0])
        subplot_6.set_array(growth[:,:,1])
        
        upper = [elem1 + elem2 for elem1, elem2 in zip(mean_temp, std_temp)]
        lower = [elem1 - elem2 for elem1, elem2 in zip(mean_temp, std_temp)]
        x = [count for count in range(len(upper))]
        subplot_2.set_data((x, mean_temp))#, color="k")
        subplot_2c.set_data((x, dead_temp))#, color="k")
        subplot_2b.set_data((x, luminosity)) #, "--", label="stellar luminosity")
        subplot_2b.axes.set_xlim(0, len(upper))
        
        subplot_2b.axes.set_xlim(0, len(upper))
        subplot_2.axes.set_xlim(0, len(upper))
        
        # population
        #print(len(pop_light), len(pop_dark))
        subplot_3.set_data((x, pop_light)) 
        subplot_4.set_data((x, pop_dark)) #, color=[0.2,0.2, 0.2], label="dark population")
        
        subplot_4.axes.set_xlim(0, len(x)) 
        
        #print(env.growth.max(), env.growth.min())
        r, grid, d, info = env.step()

    return update_fig

def get_update_fig(env, agent=None):

    global obs
    obs = env.reset()

    def update_fig_agent(ii, env=env, agent=agent):

        global subplot_0
        global subplot_1
        global subplot_2
        global subplot_3
        global subplot_4
        global subplot_5
        global subplot_6
        global subplot_2b
        
        global subplot_2_axis
        global subplot_4_axis
        global subplot_2b_axis
        global subplot_2c_axis
        global pop_dark
        global pop_light
        global mean_temp
        global std_temp
        global dead_temp
        global ax
        global ax2
        global luminosity 
        global obs

        beta = tensor_to_image(env.beta)
        temp = tensor_to_image(env.temp)
        albedo = np.clip(\
                tensor_to_image(env.grid[:,:3,:,:]),0,1.) #tensor_to_image(env.albedo)
        growth = tensor_to_image(env.growth)
        agent_grid = env.grid[:,4,:,:].squeeze()
        agent_grid = agent_grid #/ agent.max()
        
        mean_temp.append(temp.mean())
        std_temp.append(temp.std())
        dead_temp.append(env.dead_temp)
        pop_light.append(env.grid[:,1,:,:].mean())
        pop_dark.append(env.grid[:,2,:,:].mean())
        luminosity.append(env.L)
            
        subplot_0.set_array(albedo)
        subplot_1.set_array(temp)
        subplot_6.set_array(growth[:,:,1])
        
        upper = [elem1 + elem2 for elem1, elem2 in zip(mean_temp, std_temp)]
        lower = [elem1 - elem2 for elem1, elem2 in zip(mean_temp, std_temp)]
        x = [count for count in range(len(upper))]
        subplot_2.set_data((x, mean_temp))#, color="k")
        subplot_2c.set_data((x, dead_temp))#, color="k")
        
        subplot_2b.set_data((x, luminosity)) #, "--", label="stellar luminosity")
        subplot_2b.axes.set_xlim(0, len(upper))
        
        subplot_2b.axes.set_xlim(0, len(upper))
        subplot_2.axes.set_xlim(0, len(upper))
        
        # population
        #print(len(pop_light), len(pop_dark))
        subplot_3.set_data((x, pop_light)) 
        subplot_4.set_data((x, pop_dark)) #, color=[0.2,0.2, 0.2], label="dark population")
        
        subplot_4.axes.set_xlim(0, len(x)) 
        
        if agent is not None:
            subplot_5.set_array(agent_grid) #growth[:,:,0])
            action = agent(obs) #np.random.randint(4,8, size=(env.batch_size, env.n_agents, 1))
        else:
            subplot_5.set_array(growth[:,:,0])
            action = None

        obs, r, d, info = env.step(action)
        
        if d.sum() < np.zeros_like(d).sum():
            print(f"all agents done at {env.step_count}")

    return update_fig_agent
    
def seed_all(seed):
    
    np.random.seed(seed)
