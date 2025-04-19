import unittest
import numpy as np
from modules.fluid_flow import FluidFlowSimulator

class TestFluidFlowSimulator(unittest.TestCase):
    
    def setUp(self):
        # Create a simulator instance for testing
        self.simulator = FluidFlowSimulator(shape_type="circle", radius=1.0, 
                                           angle_of_attack=0.0, flow_speed=10.0)
    
    def test_initialization(self):
        # Test that the simulator initializes correctly
        self.assertEqual(self.simulator.shape_type, "circle")
        self.assertEqual(self.simulator.radius, 1.0)
        self.assertEqual(self.simulator.angle_of_attack, 0.0)
        self.assertEqual(self.simulator.flow_speed, 10.0)
    
    def test_complex_potential(self):
        # Test the complex potential calculation
        z = 2.0 + 0.0j  # Point on the x-axis
        w = self.simulator.complex_potential(z)
        
        # For a circle with radius 1 and flow along x-axis, 
        # the potential at z=2 should be U*(z + R^2/z) = 10*(2 + 1/2) = 25
        expected = 10.0 * (2.0 + 0.5)
        self.assertAlmostEqual(w.real, expected, places=5)
        self.assertAlmostEqual(w.imag, 0.0, places=5)
    
    def test_compute_velocity(self):
        # Test velocity computation
        z = 2.0 + 0.0j  # Point on the x-axis
        vel = self.simulator.compute_velocity(z)
        
        # For a circle with radius 1 and flow along x-axis,
        # the velocity at z=2 should be U*(1 - R^2/z^2) = 10*(1 - 1/4) = 7.5
        expected = 10.0 * (1.0 - 0.25)
        self.assertAlmostEqual(vel.real, expected, places=5)
        self.assertAlmostEqual(vel.imag, 0.0, places=5)
    
    def test_streamlines(self):
        # Test streamline computation
        X, Y, Psi = self.simulator.compute_streamlines()
        
        # Check dimensions
        self.assertEqual(X.shape, (self.simulator.ny, self.simulator.nx))
        self.assertEqual(Y.shape, (self.simulator.ny, self.simulator.nx))
        self.assertEqual(Psi.shape, (self.simulator.ny, self.simulator.nx))
        
        # Check symmetry for flow along x-axis
        # Psi should be antisymmetric about y=0
        mid_y = self.simulator.ny // 2
        for i in range(1, min(mid_y, 10)):  # Check a few points
            self.assertAlmostEqual(Psi[mid_y+i, self.simulator.nx//2], 
                                  -Psi[mid_y-i, self.simulator.nx//2], 
                                  places=5)

if __name__ == '__main__':
    unittest.main()