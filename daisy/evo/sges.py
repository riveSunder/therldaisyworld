import os
import json

import argparse

import time

import numpy as np
import numpy.random as npr

import sys
import subprocess
from mpi4py import MPI
comm = MPI.COMM_WORLD

from daisy.daisy_world_rl import RLDaisyWorld
from daisy.agents.greedy import Greedy
from daisy.agents.mlp import MLP

from daisy.helpers import query_kwargs

class SimpleGaussianES():

    def __init__(self, **kwargs):
        
        self.fn_dict = {"RLDaisyWorld": RLDaisyWorld,\
                "MLP": MLP}
        env_fn = RLDaisyWorld 
        self.env = env_fn(**kwargs)

        self.elitism = True
        self.champions = None
        self.leaderboard = None
        self.batch_size = self.env.batch_size
        self.max_steps = kwargs["max_steps"] if "max_steps" in kwargs.keys() else 768
        self.lr = 1.e-1
        self.number_trials = 4

        self.tag = query_kwargs("tag", "default_tag", **kwargs)
        self.seeds = query_kwargs("seeds", [42], **kwargs)
        self.entry_point = query_kwargs("entry_point", "None", **kwargs)
        # tournament bracket size
        self.bracket_size = query_kwargs("bracket_size", 5, **kwargs)
        self.num_workers = query_kwargs("num_workers", 0, **kwargs)

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

        self.population[agent_idx].reset()

        obs = self.env.reset()

        half_obs = obs.shape[1] // 2

        done = False
        all_done = False
        done_at = np.zeros((*obs.shape[:2],1), dtype=int)

        sum_reward = 0.0
        while not all_done and self.env.step_count < self.max_steps:
            
            agent = self.get_agent_action(\
                    obs[:,:half_obs], agent_idx)
            adversary = self.get_agent_action(\
                    obs[:,half_obs:], adversary_idx)

            action = np.append(agent, adversary, axis=1)

            obs, reward, done, info = self.env.step(action)
            all_done = (np.ones_like(done).sum() - done.sum()) == 0
            done_at += (1 - 1 * done)

            sum_reward += (reward[:,:half_obs]).mean()
            total_steps += (1 - 1 * done)

        sum_rewards.append(sum_reward)

        fitness = sum_reward / (obs.shape[0] * obs.shape[1])

        return fitness, total_steps, done_at.tolist()
        
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

    def mpi_fork(self):
        """
        relaunches the current script with workers
        Returns "parent" for original parent, "child" for MPI children
        (adapted from https://github.com/garymcintire/mpi_util/)
        via https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease
        """
        global num_worker, rank

        if self.num_workers <= 1:
            print("if n<=1")
            num_worker = 0
            rank = 0
            return "child"

        if os.getenv("IN_MPI") is None:
            env = os.environ.copy()
            env.update(\
                    MKL_NUM_THREADS="1", \
                    OMP_NUM_THREAdS="1",\
                    IN_MPI="1",\
                    )
            print( ["mpirun", "-np", str(self.num_workers), sys.executable] + sys.argv)
            subprocess.check_call(["mpirun", "-np", str(self.num_workers), sys.executable] \
            +['-u']+ sys.argv, env=env)

            return "parent"
        else:
            num_worker = comm.Get_size()
            rank = comm.Get_rank()
            return "child"

    def run(self, **kwargs):

        if self.mpi_fork() == "parent":
            os._exit(0)

        if rank == 0:
            self.mantle(**kwargs)
        else:
            self.arm(**kwargs)

    def mantle(self, **kwargs):

        checkpoint_every = query_kwargs("checkpoint_every", 0, **kwargs)
        max_generations = query_kwargs("max_generations", 3, **kwargs)

        t0 = time.time()

        for seed in self.seeds:
            npr.seed(seed)

            filepath = os.path.join("results", self.tag, f"{self.tag}_seed{seed}_progress.json")
            filepath_env = os.path.join("results", self.tag, f"{self.tag}_seed{seed}_daisyworld.json")
            filepath_policy = os.path.join("results", self.tag, f"{self.tag}_seed{seed}_best_agent.json")

            if os.path.exists(os.path.split(filepath)[0]):
                pass
            else:
                os.mkdir(os.path.split(filepath)[0])

            self.initialize_population()
            results = {}
            results["seed"] = seed
            results["done_at"] = []
            results["entry_point"] = query_kwargs("entry_point", "None", **kwargs)
            results["git_hash"] = query_kwargs("git_hash", "None", **kwargs)
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
                agents_done_at = []
                t1 = time.time()

                if self.num_workers <= 1:
                    for agent_idx in range(self.population_size):
                        fit = 0.0
                        agent_done_at = []
                        for trial in range(self.number_trials):
                            adversary_idx = npr.randint(self.population_size)
                            this_fitness, total_steps, done_at = self.get_fitness(agent_idx=agent_idx,\
                                    adversary_idx=adversary_idx)

                            fit += this_fitness.mean() / self.number_trials
                            total_interactions += total_steps.sum().item()
                            agent_done_at.extend(done_at)

                        fitness.append(fit)
                        agents_done_at.append(agent_done_at)
                else:
                    subpopulation_size = int(self.population_size / (self.num_workers-1))
                    population_remainder = self.population_size % (num_worker-1)
                    population_left = self.population_size

                    batch_end = 0
                    extras = 0

                    # send parameters to arms
                    for cc in range(1, self.num_workers):
                        run_batch_size = min(subpopulation_size, population_left)

                        if population_remainder:
                            run_batch_size += 1
                            population_remainder -= 1
                            extras += 1

                        batch_start = batch_end 
                        batch_end = batch_start + run_batch_size 

                        parameters_list = [my_agent.get_parameters() \
                                for my_agent in self.population]

                        agent_indices = [elem for elem in range(batch_start, batch_end)]

                        comm.send((parameters_list, agent_indices), dest=cc)

                    # receive current generation's fitnesses from arm processes
                    for dd in range(1, num_worker):
                        fit, total_steps, agent_done_at = comm.recv(source=dd)

                        fitness.extend(fit)

                        agents_done_at.extend(agent_done_at)
                        
                        total_interactions += total_steps

                self.update_population(fitness)

                t2 = time.time()
                # numpy ndarrays cannot be serialized to json. convert to list first
                results["done_at"].append(agents_done_at)
                results["wall_time"].append(t2-t0)
                results["generation"].append(generation)
                results["total_interactions"].append(total_interactions)
                results["mean_fitness"].append(np.mean(fitness))
                results["variance_fitness"].append(np.var(fitness))
                results["min_fitness"].append(np.min(fitness))
                results["max_fitness"].append(np.max(fitness))

                # don't save anything if checkpoint_every is 0
                if checkpoint_every:
                    if generation % checkpoint_every == 0 or generation == (max_generations-1):
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
                                f"{self.tag}_seed{seed}_best_agent_gen{generation}.json")

                        self.population[0].save_config(filepath_policy)

                        filepath_numpy_pop =  os.path.join("results", self.tag, \
                                f"{self.tag}_seed{seed}_population_gen{generation}.npy")

                        population_params = self.population[0].get_parameters()[None,:]

                        for ii in range(1, len(self.population)):
                            my_params = self.population[ii].get_parameters()[None,:]
                            population_params = np.append(population_params, my_params) 

                        np.save(filepath_numpy_pop, population_params)

        for ee in range(1, self.num_workers):
            print(f"send shutown signal to worker {ee}")
            comm.send((0, 0), dest=ee)

    def arm(self, **kwargs):

        while True:

            parameters_list, agent_indices = comm.recv(source=0)
            if parameters_list == 0:
                print(f"worker {rank} shutting down")
                break

            self.population_size = len(parameters_list)

            for ff in range(self.population_size):
                self.population[ff].set_parameters(parameters_list[ff])
                

            fitness = []
            agents_done_at = []
            total_interactions = 0
            for agent_idx in agent_indices:
                fit = 0.0
                agent_done_at = []
                for trial in range(self.number_trials):
                    adversary_idx = npr.randint(self.population_size)
                    this_fitness, total_steps, done_at = self.get_fitness(agent_idx=agent_idx,\
                            adversary_idx=adversary_idx)

                    fit += this_fitness.mean() / self.number_trials
                    total_interactions += total_steps.sum().item()
                    agent_done_at.extend(done_at)

                fitness.append(fit)
                agents_done_at.append(agent_done_at)

            comm.send((fitness, total_interactions, agents_done_at), dest=0)
 
    
    def plot_run(self, logs=None):
        pass

    def save_population(self, filepath="./default_pop.npy"):
        pass

    def load_population(self, filepath="./default_pop.npy"):
        pass

if __name__ == "__main__": #pragma: no cover


    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--checkpoint_every", type=int, default=16,\
            help="saving checkpoint every so often")
    parser.add_argument("-d", "--grid_dimension", type=int, default=16,\
            help="length of each side for daisyworld grid")
    parser.add_argument("-g", "--max_generations", type=int, default=16,\
            help="number of generations to evolve")
    parser.add_argument("-p", "--population_size", type=int, default=16,\
            help="number of individuals in the population")
    parser.add_argument("-s", "--seeds", type=int, nargs="+", default=[42],\
            help="seeds for pseudo-random number generator")
    parser.add_argument("-t", "--tag", type=str, default="cmaes_tag",\
            help="tag for identifying experiment")
    parser.add_argument("-w", "--num_workers", type=int, default=0,\
            help="number of workers (arm processes), not including mantle process")

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    hash_command = ["git", "rev-parse", "--verify", "HEAD"]
    git_hash = subprocess.check_output(hash_command)

    # store the command-line call for this experiment
    entry_point = []
    entry_point.append(os.path.split(sys.argv[0])[1])
    args_list = sys.argv[1:]

    sorted_args = []
    for aa in range(0, len(args_list)):

        if "-" in args_list[aa]:
            sorted_args.append([args_list[aa]])
        else: 
            sorted_args[-1].append(args_list[aa])

    sorted_args.sort()
    entry_point = "python -m daisy.evo.sges"

    for elem in sorted_args:
        entry_point += " " + " ".join(elem)

    kwargs["entry_point"] = entry_point 
    kwargs["git_hash"] = git_hash.decode("utf8")[:-1]
    evo = SimpleGaussianES(**kwargs) 
    evo.run(**kwargs)
