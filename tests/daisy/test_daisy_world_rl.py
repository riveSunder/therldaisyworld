import unittest
import numpy as np

import numpy.random as npr

from daisy.daisy_world_rl import RLDaisyWorld


class TestRLDaisyWorld(unittest.TestCase):

  def setUp(self):
        pass

  def test_rl_daisy_world_ad_hoc(self):

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

  def test_temp(self):
    """
    test to make sure env.grid keeps track of temp states
    """
    env = RLDaisyWorld()

    self.assertLess(0, env.grid[:,3].mean())
    self.assertLess(0, env.grid[:,4].mean())
    self.assertLess(0, env.grid[:,5].mean())

    o = env.reset()
    
    self.assertLess(0, env.grid[:,3].mean())
    self.assertLess(0, env.grid[:,4].mean())
    self.assertLess(0, env.grid[:,5].mean())

    obs, reward, done, info = env.step()

    self.assertLess(0, env.grid[:,3].mean())
    self.assertLess(0, env.grid[:,4].mean())
    self.assertLess(0, env.grid[:,5].mean())
    self.assertLess(0, obs[:,:,3].mean())
    self.assertLess(0, obs[:,:,4].mean())
    self.assertLess(0, obs[:,:,5].mean()) 

    action = npr.randint(9, size=(env.batch_size, env.n_agents, 1))
    obs, reward, done, info = env.step(action)

    self.assertLess(0, env.grid[:,3].mean())
    self.assertLess(0, env.grid[:,4].mean())
    self.assertLess(0, env.grid[:,5].mean())
    self.assertLess(0, obs[:,:,3].mean())
    self.assertLess(0, obs[:,:,4].mean())
    self.assertLess(0, obs[:,:,5].mean()) 


if __name__ == "__main__": #pragma: no cover

    unittest.main(verbosity=2)
