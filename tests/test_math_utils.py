import unittest
import numpy as np
from modules.math_utils import (verify_cauchy_riemann, compute_taylor_series, 
                              compute_laurent_series, compute_residue, 
                              solve_eigenvalue_problem)

class TestMathUtils(unittest.TestCase):
    
    def test_verify_cauchy_riemann(self):
        # Test with an analytic function (e^z)
        def f(z):
            return np.exp(z)
        
        is_satisfied, u_x, v_y, u_y, v_x = verify_cauchy_riemann(f, 1+1j)
        
        # e^z satisfies Cauchy-Riemann
        self.assertTrue(is_satisfied)
        
        # For e^z, u_x = v_y = e^x * cos(y) and u_y = -v_x = e^x * sin(y)
        expected_u_x = np.exp(1) * np.cos(1)
        expected_v_y = np.exp(1) * np.cos(1)
        expected_u_y = np.exp(1) * np.sin(1)
        expected_v_x = -np.exp(1) * np.sin(1)
        
        self.assertAlmostEqual(u_x, expected_u_x, places=5)
        self.assertAlmostEqual(v_y, expected_v_y, places=5)
        self.assertAlmostEqual(u_y, expected_u_y, places=5)
        self.assertAlmostEqual(v_x, expected_v_x, places=5)
    
    def test_compute_taylor_series(self):
        # Test with a simple function (e^z)
        def f(z):
            return np.exp(z)
        
        coeffs = compute_taylor_series(f, 0, order=5)
        
        # Taylor series of e^z around 0 is 1 + z + z^2/2! + z^3/3! + ...
        expected_coeffs = [1, 1, 1/2, 1/6, 1/24, 1/120]
        
        for i, coeff in enumerate(coeffs):
            self.assertAlmostEqual(coeff, expected_coeffs[i], places=5)
    
    def test_compute_laurent_series(self):
        # Test with a function having a pole (1/z)
        def f(z):
            return 1/z
        
        coeffs = compute_laurent_series(f, 0, inner_order=3, outer_order=3)
        
        # Laurent series of 1/z around 0 has only a_-1 = 1, all others are 0
        expected_idx = 3 - 1  # Index of a_-1 in the coefficients list
        
        for i, coeff in enumerate(coeffs):
            if i == expected_idx:
                self.assertAlmostEqual(abs(coeff), 1.0, places=1)
            else:
                self.assertAlmostEqual(abs(coeff), 0.0, places=1)
    
    def test_compute_residue(self):
        # Test with a function having a simple pole (1/z)
        def f(z):
            return 1/z
        
        residue = compute_residue(f, 0)
        
        # Residue of 1/z at z=0 is 1
        self.assertAlmostEqual(residue, 1.0, places=1)
    
    def test_solve_eigenvalue_problem(self):
        # Test with a simple matrix
        A = np.array([[2, 1], [1, 3]])
        
        eigenvalues, eigenvectors = solve_eigenvalue_problem(A)
        
        # Eigenvalues of [[2, 1], [1, 3]] are approximately 1.38 and 3.62
        expected_eigenvalues = [1.38, 3.62]
        
        self.assertAlmostEqual(eigenvalues[0], expected_eigenvalues[0], places=2)
        self.assertAlmostEqual(eigenvalues[1], expected_eigenvalues[1], places=2)
        
        # Check that eigenvectors are normalized
        for i in range(2):
            self.assertAlmostEqual(np.linalg.norm(eigenvectors[:, i]), 1.0, places=5)
        
        # Check that A*v = lambda*v
        for i in range(2):
            Av = np.dot(A, eigenvectors[:, i])
            lambda_v = eigenvalues[i] * eigenvectors[:, i]
            self.assertTrue(np.allclose(Av, lambda_v, rtol=1e-5))

if __name__ == '__main__':
    unittest.main()