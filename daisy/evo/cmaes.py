import numpy as np

import numpy.random as npr

from mpi4py import MPI
comm = MPI.COMM_WORLD

from daisy.daisy_world_rl import RLDaisyWorld
from daisy.agents.greedy import Greedy
from daisy.agents.mlp import MLP

from daisy.helpers import query_kwargs

from daisy.evo.sges import SimpleGaussianES

class CMAES(SimpleGaussianES):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def calculate_stats(self, population, elite_population=None):
        

        if elite_population is not None:
            pop_params = elite_population[0].get_parameters()[None]

            for member in elite_population[1:]:
                
                pop_params = np.append(pop_params,\
                        member.get_parameters()[None,:], axis=0)

            my_mean = np.mean(pop_params, axis=0, keepdims=True)
            my_covariance = np.matmul((my_mean - self.mean).T, (my_mean-self.mean))

        else:

            pop_params = population[0].get_parameters()[None]
            for member in population[1:]:
                
                pop_params = np.append(pop_params,\
                        member.get_parameters()[None,:], axis=0)

            my_mean = np.mean(pop_params, axis=0, keepdims=True)

            my_covariance = np.matmul((my_mean).T, (my_mean))

        return my_mean, my_covariance

    def initialize_population(self):

        self.population = [self.agent_fn(**self.agent_args) \
                for ii in range(self.population_size)]

        self.mean, self.covariance = self.calculate_stats(\
                self.population)

    def update_population(self, fitness):

        sorted_indices = list(np.argsort(fitness))
        sorted_indices.reverse()
        
        elite_pop = np.array(self.population)\
                [sorted_indices[0:self.keep_elite]]

        elite_mean, covariance = self.calculate_stats(population=0, \
                elite_population=elite_pop)

        self.mean = (1. - self.lr)* self.mean + self.lr*elite_mean

        self.covariance = (1. - self.lr) \
                * self.covariance \
                + self.lr * covariance 

        for ii in range(self.population_size):
            
            if self.elitism and ii < self.keep_elite:

                self.population[ii].set_parameters(\
                        self.population[\
                        sorted_indices[ii]].get_parameters())
            else:

                new_parameters = npr.multivariate_normal(self.mean.squeeze(), \
                        self.covariance)

                new_parameters = new_parameters.ravel()

                self.population[ii].set_parameters(new_parameters)


if __name__ == "__main__":

    evo = CMAES()
    evo.run(max_generations=10)
