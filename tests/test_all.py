import unittest

from tests.daisy.test_daisy_world_rl import TestRLDaisyWorld
from tests.daisy.evo.test_cmaes import TestCMAES
from tests.daisy.evo.test_sges import TestSimpleGaussianES
from tests.daisy.agents.test_greedy import TestGreedy

if __name__ == "__main__":

    unittest.main(verbosity=2)
