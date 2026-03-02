import unittest
import numpy as np
from simulate import Ball
from constants import *

class TestBasketballPhysics(unittest.TestCase):
    
    def test_energy_conservation(self):
        """ Test that a ball dropped from height does not gain energy after a ground bounce. """
        # Drop ball from 10ft with 0 initial velocity
        ball = Ball(0, 0, 10, 1e-6, 0, 0, 0)
        
        # Find the first bounce
        found_bounce = False
        v_prev_max = 0
        for i in range(1, len(ball.states)):
            z = ball.states[i][2]
            vz = ball.states[i][5]
            if vz < 0:
                v_prev_max = max(v_prev_max, np.abs(vz))
            if vz > 0 and v_prev_max > 1.0: # Significant bounce detected
                found_bounce = True
                # Energy after bounce should be less than or equal to energy before
                self.assertLessEqual(np.abs(vz), v_prev_max + 1e-6)
                break
        self.assertTrue(found_bounce, "Ball did not bounce on the ground")

    def test_known_swish(self):
        """ Test that a perfect shot is correctly detected as a score. """
        # A known high-probability scoring shot
        ball = Ball(15, 0, 6, 26, 56, 0, 5)
        self.assertTrue(ball.score, "Perfect shot failed to score")

    def test_out_of_bounds(self):
        """ Test that shots far away are correctly terminated and marked as missed. """
        # Shot launched away from the rim
        ball = Ball(15, 0, 6, 26, 56, 180, 5)
        self.assertFalse(ball.score)
        self.assertTrue(ball.end)

if __name__ == '__main__':
    unittest.main()
