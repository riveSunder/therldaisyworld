import unittest
import numpy as np

from daisy.daisy_world_rl import RLDaisyWorld


class TestRLDaisyWorld(unittest.TestCase):

    def setUp(self):
        pass

    def test_ad_hoc(self):

        env = RLDaisyWorld()

        a = env.grid
        b = env.forward(a)

        for ii in range(9):
            for jj in range(1):
                action = np.array([[[ii]]]) #randint(9, size=(env.batch_size, env.n_agents, 1))

                obs, reward, done, info = env.step(action)


        self.assertFalse(done.mean())
        self.assertTrue(type(info) == dict)
        self.assertLessEqual(0.0, reward.mean())
        self.assertTrue(a.shape == b.shape)
        self.assertTrue(obs.shape[1] == env.n_agents)
        self.assertTrue(obs.shape[0] == env.batch_size)

if __name__ == "__main__": #pragma: no cover

    unittest.main(verbosity=2)
