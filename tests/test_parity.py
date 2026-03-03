import unittest
import numpy as np
from vectorized_analysis import run_analysis
from simulate import Ball

class TestVectorizedParity(unittest.TestCase):
    def test_parity_5x5(self):
        """
        Verify that vectorized_analysis results match individual simulate.Ball shots.
        """
        # 1. Run low-res vectorized analysis (5x5 grid = 25 shots)
        # speeds: [25, 26.5, 28, 29.5, 31], phis: [37, 44.75, 52.5, 60.25, 68]
        nx, ny = 5, 5
        speeds, phis, scored_map = run_analysis(nx=nx, ny=ny, save=False, plot=False)
        
        print("\nChecking Parity for 5x5 grid...")
        
        # 2. Compare each point in the grid with a single Ball simulation
        for i in range(ny): # Angle index
            for j in range(nx): # Speed index
                v = speeds[j]
                phi = phis[i]
                
                # Run single-shot simulator
                # (Free throw pos: 15, 0, 6; spin: 5)
                ball = Ball(15, 0, 6, v, phi, 0, 5)
                
                expected = bool(scored_map[i, j])
                actual = ball.score
                
                with self.subTest(speed=v, angle=phi):
                    self.assertEqual(expected, actual, 
                        f"Mismatch at v={v:.2f}, phi={phi:.2f}. Vectorized: {expected}, Single: {actual}")

if __name__ == '__main__':
    unittest.main()
