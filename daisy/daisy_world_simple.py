import numpy as np

import matplotlib.pyplot as plt

class SimpleDaisyWorld():

    def __init__(self, **kwargs):
        
        self.my_cmap = plt.get_cmap("magma")
        self.my_cmap2 = plt.get_cmap("viridis")
        # model parameters
        self.p = 1.0
        self.g = 0.003265
        self.S = 1000.0
        self.sigma = 5.67e-8
        self.gamma = 0.05
        self.q = 0.2 * self.S / self.sigma
        self.Toptim = 295.5
        self.dt = 0.01
        #albedos
        self.Ag = 0.5
        self.Aw = 0.75
        self.Ab = 0.25

        self.max_L = 2.00
        self.min_L = 0.7
        self.steps_per_period = 10000

        self.initial_L = self.min_L
        self.initial_ab = 0.2
        self.initial_aw = 0.2
        self.initial_ag = self.p - self.initial_ab - self.initial_aw

        self.reset()


    def reset(self):

        self.dL = 2 * (self.max_L - self.min_L) / self.steps_per_period

        self.ag = self.initial_ag
        self.ab = self.initial_ab
        self.aw = self.initial_aw

        self.L = self.initial_L

        self.steps = 0
        self.list_A = []
        self.list_Te = []
        self.list_Tg = []
        self.list_Tb = []
        self.list_Tw = []
        self.list_T_lifeless = []
        self.list_L = []
        self.list_beta_b = []
        self.list_beta_w = []
        self.list_dab_dt = []
        self.list_daw_dt = []
        self.list_ab = []
        self.list_aw = []
        self.list_ag = []
        self.list_steps = []

    def update_L(self, L):
        
        if self.steps % self.steps_per_period == 0:
            self.dL *= -1
        
        return max([min([self.max_L, L + self.dL]), self.min_L])

    def step(self):

        # global albedo
        self.A = self.ag * self.Ag + self.aw * self.Aw + self.ab * self.Ab
        # effective global temperature
        self.Te = ((self.S * self.L * (1.0 - self.A))/self.sigma)**(1/4.)
        self.T_lifeless = ((self.S * self.L * (1.0 - self.Ag))/self.sigma)**(1/4.)
        # local temperature, bare ground, and white and black daisies
        self.Tg = (self.q * (self.A - self.Ag) + self.Te**4.)**(1/4.)
        self.Tb = (self.q * (self.A - self.Ab) + self.Te**4.)**(1/4.)
        self.Tw = (self.q * (self.A - self.Aw) + self.Te**4.)**(1/4.)
        #print(self.Tb, self.Tg, self.Tw)
        # daisy growth rates
        self.beta_b = 1 - self.g * (self.Toptim - self.Tb)**2
        self.beta_w = 1 - self.g * (self.Toptim - self.Tw)**2
        # change in daisy populations
        self.dab_dt = self.ab * (self.ag*self.beta_b - self.gamma)
        self.daw_dt = self.aw * (self.ag*self.beta_w - self.gamma)
        # ground covered by daisies
        self.ab = self.ab + self.dt * self.dab_dt
        self.aw = self.aw + self.dt * self.daw_dt
        # bare, suitable land not populated by daisies
        self.ag = self.p - self.aw - self.ab 

        self.steps += 1
        self.L = self.update_L(self.L)

    def store_values(self):
        
        self.list_A.append(self.A)
        self.list_Te.append(self.Te)
        self.list_Tg.append(self.Tg)
        self.list_Tb.append(self.Tb)
        self.list_Tw.append(self.Tw)
        self.list_T_lifeless.append(self.T_lifeless)
        self.list_beta_b.append(self.beta_b)
        self.list_beta_w.append(self.beta_w)
        self.list_dab_dt.append(self.dab_dt)
        self.list_daw_dt.append(self.daw_dt)
        self.list_ab.append(self.ab)
        self.list_aw.append(self.aw)
        self.list_ag.append(self.ag)
        self.list_steps.append(self.steps)
        self.list_L.append(self.L)

    def run_sim(self, num_periods=1):

        for period in range(num_periods):
            for step in range(self.steps_per_period):
                self.step()
                self.store_values()

    def plot_curve(self, show_habitable=False):

        fig, ax = plt.subplots(2,1, figsize=(10,8))
        ax2 = ax[1].twinx()

        lines = ax[1].plot(self.list_steps, self.list_L, "--", \
                color=[0.9,0.9,0.6], label="Stellar Luminosity", lw=5, alpha=0.5)
        lines += ax2.plot(self.list_steps, self.list_T_lifeless, \
                color=[0.1,0.1,0.1], label="lifeless temp.", lw=5, alpha=0.5)
        lines += ax2.plot(self.list_steps, self.list_Te, \
                color=self.my_cmap2(128), label="daisyworld temp", lw=5, alpha=0.5)

        if show_habitable:
            pm_range = np.sqrt(1 / self.g)

            my_x = [0, max(self.list_steps)]
            upper = self.Toptim + pm_range
            lower = self.Toptim - pm_range

            my_fill = ax2.fill_between(my_x, [lower, lower],\
                    [upper, upper], alpha=0.1225, color=self.my_cmap2(96),\
                    label="habitable range")
            lines += ax2.plot(my_x, [self.Toptim, self.Toptim],\
                    alpha=0.1225, color=self.my_cmap2(96), \
                    label="habitable range")

        labels = [line.get_label() for line in lines]
        ax[1].legend(lines, labels, loc=2)
        ax[0].plot(self.list_steps, self.list_ab, "-.", \
                color=self.my_cmap(0), label="black daisies", lw=5, alpha=0.5)
        ax[0].plot(self.list_steps, self.list_aw, "--", \
                color=self.my_cmap(200), label="white daisies", lw=5, alpha=0.5)
        ax[0].legend()

        ax[1].set_xlim(0, max(self.list_steps))
        ax[0].set_xlim(0, max(self.list_steps))



        return fig, ax

if __name__ == "__main__":

    world = SimpleDaisyWorld()

    world.run_sim()
    fig, ax = world.plot_curve()
    fig.savefig("assets/daisy_world_simple.png")


