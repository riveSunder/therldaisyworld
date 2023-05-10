import os
import sys
import multiprocessing
import subprocess

import unittest
import numpy as np

from daisy.evo.cmaes import CMAES


class TestCMAES(unittest.TestCase):

    def setUp(self):
        pass

    def test_cmaes_ad_hoc(self):

        num_threads = multiprocessing.cpu_count()

        kwargs = {}

        kwargs["checkpoint_every"] = 0
        kwargs["grid_dimension"] = 16
        kwargs["max_generations"] = 2
        kwargs["population_size"] = 8
        kwargs["seeds"] = [42]
        kwargs["tag"] = "testing_run"
        kwargs["num_workers"] = 0
        kwargs["max_steps"] = 10
        kwargs["batch_size"] = 4

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
        entry_point = "python -m daisy.evo.cmaes"

        for elem in sorted_args:
            entry_point += " " + " ".join(elem)

        kwargs["entry_point"] = entry_point 
        kwargs["git_hash"] = git_hash.decode("utf8")[:-1]

        evo = CMAES(**kwargs)
        evo.run(**kwargs)

        self.assertTrue(True)

if __name__ == "__main__": #pragma: no cover

    unittest.main(verbosity=2)

