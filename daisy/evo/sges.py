import os
import json

import argparse

import time

import numpy as np
import numpy.random as npr

#from mpi4py import MPI
#comm = MPI.COMM_WORLD

from daisy.daisy_world_rl import RLDaisyWorld
from daisy.agents.greedy import Greedy
from daisy.agents.mlp import MLP

from daisy.helpers import query_kwargs

class SimpleGaussianES():

    def __init__(self, **kwargs):
        
        self.fn_dict = {"RLDaisyWorld": RLDaisyWorld,\
                "MLP": MLP}
        env_fn = RLDaisyWorld 
        self.env = env_fn()

        self.elitism = True
        self.champions = None
        self.leaderboard = None
        self.batch_size = self.env.batch_size
        self.max_steps = 768 #env.ramp_period * 3
        self.lr = 1.e-1

        #
        self.tag = query_kwargs("tag", "default_tag", **kwargs)
        self.entry_point = query_kwargs("entry_point", "None", **kwargs)
        # tournament bracket size
        self.bracket_size = query_kwargs("bracket_size", 5, **kwargs)
        self.num_workers = query_kwargs("num_workers", 1, **kwargs)

        self.population_size = query_kwargs("population_size", \
                16, **kwargs)
        self.keep_elite = max([self.population_size // 8,1])

        self.agent_fn = query_kwargs("agent_fn", MLP, **kwargs)

        self.agent_args = {"None": None}

        self.initialize_population()

    def make_config(self):

        config = {}
        config["tag"] = self.tag
        config["env_fn"] = self.env.__class__.__name__
        config["elitism"] = self.elitism
        config["batch_size"] = self.batch_size
        config["max_steps"] = self.max_steps
        config["lr"] = self.lr
        config["entry_point"] = self.entry_point
        config["bracket_size"] = self.bracket_size
        config["num_workers"] = self.num_workers
        config["population_size"] = self.population_size
        config["keep_elite"] = self.keep_elite
        config["agent_fn"] = self.population[0].__class__.__name__

        return config

    def _apply_config(self, config):

        self.tag = config["tag"]
        self.env_fn = self.fn_dict[config["env_fn"]]
        self.elitism = config["elitism"]
        self.batch_size = config["batch_size"] 
        self.max_steps = config["max_steps"] 
        self.lr = config["lr"] 
        self.entry_point = config["entry_point"] 
        self.bracket_size = config["bracket_size"] 
        self.num_workers = config["num_workers"] 
        self.population_size = config["population_size"] 
        self.keep_elite = config["keep_elite"] 
        self.agent_fn = self.fn_dict[config["agent_fn"]]

    def save_config(self, filepath=None):

        if filepath is None:
            filepath = os.path.join("results", "default_exp_config.json")

        config = self.make_config()

        with open(filepath, "w") as f:
            json.dump(config, f)


    def load_config(self, filepath=None):

        if filepath is None:
            filepath = os.path.join("results", "default_exp_config.json")

        with open(filepath, "r") as f:
            config = json.load(f)

        return config

    def restore_config(self, filepath=None):

        if filepath is None:
            filepath = os.path.join("results", "default_exp_config.json")

        config = self.load_config(filepath)

        self._apply_config(config)

    def calculate_stats(self, population):
        
        pop_params = population[0].get_parameters()[None]

        for member in population[1:]:
            
            pop_params = np.append(pop_params,\
                    member.get_parameters()[None,:], axis=0)

        my_mean = np.mean(pop_params, axis=0, keepdims=True)
        my_standard_deviation = np.std(pop_params, axis=0, \
                keepdims=True)

        return my_mean, my_standard_deviation
        
    def initialize_population(self):
        self.population = [self.agent_fn(**self.agent_args) \
                for ii in range(self.population_size)]

        self.mean, self.standard_deviation = self.calculate_stats(\
                self.population)

    def get_agent_action(self, obs, agent_idx=0):
        return self.population[agent_idx].get_action(obs)

    def get_fitness(self, agent_idx=0, adversary_idx=0): 

        fitness = []
        sum_rewards = []
        total_steps = 0
        self.env.ramp_period = 128

        self.population[agent_idx].reset()

        obs = self.env.reset()

        half_obs = obs.shape[1] // 2

        done = False
        sum_reward = 0.0
        while not done and self.env.step_count < self.max_steps:
            
            agent = self.get_agent_action(\
                    obs[:,:half_obs], agent_idx)
            adversary = self.get_agent_action(\
                    obs[:,half_obs:], adversary_idx)

            action = np.append(agent, adversary, axis=1)

            obs, reward, done, info = self.env.step(action)
            done = (np.ones_like(done).sum() - done.sum()) == 0
#            if done: 
#                print(f"done t step {self.env.step_count}")
#                print(self.env.grid[:,1:3].sum(axis=(-3,-2,-1)))
#                print(reward)

            sum_reward += (reward[:,:half_obs]).mean()
            total_steps += 1

        sum_rewards.append(sum_reward)

        fitness = sum_reward / (obs.shape[0] * obs.shape[1])

        return fitness, total_steps
        
    def update_population(self, fitness):

        sorted_indices = list(np.argsort(fitness))
        sorted_indices.reverse()
        
        elite_pop = np.array(self.population)\
                [sorted_indices[0:self.keep_elite]]

        elite_mean, elite_standard_deviation = self.calculate_stats(\
                elite_pop)

        self.mean = (1. - self.lr)* self.mean + self.lr*elite_mean
        self.standard_deviaton = (1. - self.lr) \
                * self.standard_deviation \
                + self.lr * elite_standard_deviation

        for ii in range(self.population_size):
            
            if self.elitism and ii < self.keep_elite:

                self.population[ii].set_parameters(\
                        self.population[\
                        sorted_indices[ii]].get_parameters())
            else:

                new_parameters = npr.randn(*self.mean.shape) \
                        * self.standard_deviation \
                        + self.mean
                new_parameters = new_parameters.ravel()

                self.population[ii].set_parameters(new_parameters)


    def run(self, max_generations=10, **kwargs):

        number_trials = 4

        checkpoint_every = query_kwargs("checkpoint_every", 1, **kwargs)

        t0 = time.time()

        filepath = os.path.join("results", self.tag, f"{self.tag}_progress.json")
        filepath_env = os.path.join("results", self.tag, f"{self.tag}_daisyworld.json")
        filepath_policy = os.path.join("results", self.tag, f"{self.tag}_best_agent.json")

        if os.path.exists(os.path.split(filepath)[0]):
            pass
        else:
            os.mkdir(os.path.split(filepath)[0])

        results = {}
        results["wall_time"] = []
        results["generation"] = []
        results["total_interactions"] = []
        results["mean_fitness"] = []
        results["variance_fitness"] = []
        results["min_fitness"] = []
        results["max_fitness"] = []

        # total number of interactions with the environment
        total_interactions = 0
        for generation in range(max_generations):

            fitness = []    
            t1 = time.time()
            for agent_idx in range(self.population_size):
                fit = 0.0
                for trial in range(number_trials):
                    adversary_idx = npr.randint(self.population_size)
                    this_fitness, total_steps = self.get_fitness(agent_idx=agent_idx,\
                            adversary_idx=adversary_idx)

                    fit += this_fitness.mean() / number_trials
                    total_interactions += total_steps

                fitness.append(fit)

            self.update_population(fitness)

            t2 = time.time()
            results["wall_time"].append(t2-t0)
            results["generation"].append(generation)
            results["total_interactions"].append(total_interactions)
            results["mean_fitness"].append(np.mean(fitness))
            results["variance_fitness"].append(np.var(fitness))
            results["min_fitness"].append(np.min(fitness))
            results["max_fitness"].append(np.max(fitness))

            if generation % checkpoint_every == 0:
                # save progress
                elapsed = t2 - t0
                elapsed_generation = t2 - t1

                msg = f"generation {generation}, {results['wall_time'][-1]:.0f} s elapsed "
                msg += f"mean fitness +/- std. deviation: {results['mean_fitness'][-1]:.1e} +/- "
                msg += f"{np.sqrt(results['variance_fitness'][-1]):.1e}, "
                msg += f"max: {results['max_fitness'][-1]:.1e} "
                msg += f"min: {results['min_fitness'][-1]:.1e}"

                print(msg)

                with open(filepath, "w") as f:
                    json.dump(results, f)

                if generation == 0:
                    self.env.save_config(filepath_env)

                filepath_policy = os.path.join("results", self.tag, \
                        f"{self.tag}_best_agent_gen{generation}.json")
                self.population[0].save_config(filepath_policy)

                filepath_numpy_pop =  os.path.join("results", self.tag, \
                        f"{self.tag}_population_gen{generation}.npy")

                population_params = self.population[0].get_parameters()[None,:]

                for ii in range(1, len(self.population)):
                    my_params = self.population[ii].get_parameters()[None,:]
                    population_params = np.append(population_params, my_params) 

                np.save(filepath_numpy_pop, population_params)

 
    
    def plot_run(self, logs=None):
        pass

    def save_population(self, filepath="./default_pop.npy"):
        pass

    def load_population(self, filepath="./default_pop.npy"):
        pass
        

if __name__ == "__main__":


    algo = SimpleGaussianES(population_size=16) 

    algo.run(max_generations=3)
