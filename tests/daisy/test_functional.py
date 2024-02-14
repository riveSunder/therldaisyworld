import unittest
import numpy as np

from daisy.daisy_world_rl import RLDaisyWorld

from daisy.nn.functional import make_neighborhood, \
        make_von_neumann,\
        make_moore,\
        make_circular


class TestFunctional(unittest.TestCase):

    def setUp(self):
        pass

    def test_neighborhoods(self):

        for mode in ["moore", "von_neumann", "circular", "asdf"]:

            for kr in np.arange(1,5):

                expected_dim = 2*kr+1

                nbhd = make_neighborhood(radius=kr, mode=mode)

                self.assertEqual(expected_dim, nbhd.shape[-2])
                self.assertEqual(expected_dim, nbhd.shape[-1])

                #center value is always 1
                self.assertTrue(nbhd[kr,kr] == 1)

                if mode == "moore":
                    # corners are included
                    nbhd[0,0] == 1.0
                    nbhd[-1,0] == 1.0
                    nbhd[0,-1] == 1.0
                    nbhd[-1,-1] == 1.0
                else:
                    #no corners
                    nbhd[0,0] == 0.0
                    nbhd[-1,0] == 0.0
                    nbhd[0,-1] == 0.0
                    nbhd[-1,-1] == 0.0


