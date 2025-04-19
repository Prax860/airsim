import unittest
import numpy as np
from modules.vibration import VibrationAnalyzer

class TestVibrationAnalyzer(unittest.TestCase):
    
    def setUp(self):
        # Create an analyzer instance for testing
        self.analyzer = VibrationAnalyzer(structure_type="mass_spring", n_masses=3, 
                                         spring_constant=10.0, mass_value=1.0)
    
    def test_initialization(self):
        # Test that the analyzer initializes correctly
        self.assertEqual(self.analyzer.structure_type, "mass_spring")
        self.assertEqual(self.analyzer.n_masses, 3)
        self.assertEqual(self.analyzer.spring_constant, 10.0)
        self.assertEqual(self.analyzer.mass_value, 1.0)
    
    def test_matrix_building(self):
        # Test that matrices are built correctly
        # Mass matrix should be diagonal with mass values
        self.assertTrue(np.allclose(self.analyzer.M, np.eye(3)))
        
        # Stiffness matrix for 3-mass system should be:
        # [2k, -k, 0; -k, 2k, -k; 0, -k, 2k]
        expected_K = np.array([
            [20.0, -10.0, 0.0],
            [-10.0, 20.0, -10.0],
            [0.0, -10.0, 20.0]
        ])
        self.assertTrue(np.allclose(self.analyzer.K, expected_K))
    
    def test_compute_modes(self):
        # Test mode computation
        frequencies, modes = self.analyzer.compute_modes()
        
        # Check dimensions
        self.assertEqual(len(frequencies), 3)
        self.assertEqual(modes.shape, (3, 3))
        
        # Check that frequencies are positive and sorted
        self.assertTrue(np.all(frequencies > 0))
        self.assertTrue(np.all(np.diff(frequencies) >= 0))
        
        # Check orthogonality of modes with respect to mass matrix
        for i in range(3):
            for j in range(i+1, 3):
                dot_product = np.dot(modes[:, i], np.dot(self.analyzer.M, modes[:, j]))
                self.assertAlmostEqual(dot_product, 0.0, places=5)
    
    def test_latex_representation(self):
        # Test LaTeX representation of matrices
        mass_latex = self.analyzer.get_mass_matrix_latex()
        stiffness_latex = self.analyzer.get_stiffness_matrix_latex()
        
        # Check that the LaTeX strings are non-empty
        self.assertTrue(len(mass_latex) > 0)
        self.assertTrue(len(stiffness_latex) > 0)
        
        # Check that they contain the matrix elements
        self.assertIn("1.0", mass_latex)
        self.assertIn("20.0", stiffness_latex)
        self.assertIn("-10.0", stiffness_latex)

if __name__ == '__main__':
    unittest.main()