import os
import sys
import multiprocessing
import subprocess

import unittest
import numpy as np

from daisy.evo.cmaes import CMAES
from daisy.agents.greedy import Greedy
from daisy.daisy_world_rl import RLDaisyWorld

class TestGreedy(unittest.TestCase):

    def setUp(self):
        pass

    def test_greedy_ad_hoc(self):

        kwargs = {"max_steps": 50}

        env = RLDaisyWorld(**kwargs)
        obs = env.reset()

        agent = Greedy()

        greedy_sum = 0.0
        for ii in range(env.ramp_period):
            action = agent(obs)
            obs, r, d, i = env.step(action)

            greedy_sum += r.sum().item()

        self.assertTrue(True)

if __name__ == "__main__": #pragma: no cover

    unittest.main(verbosity=2)
