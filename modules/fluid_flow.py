import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sympy as sp

class FluidFlowSimulator:
    """
    Class for simulating fluid flow around various shapes using complex potential theory.
    """
    
    def __init__(self, shape_type="circle", radius=1.0, naca_code=None, 
                 angle_of_attack=0.0, flow_speed=10.0):
        """
        Initialize the fluid flow simulator.
        
        Parameters:
        -----------
        shape_type : str
            Type of shape to simulate flow around ('circle', 'airfoil', 'custom')
        radius : float
            Radius of the circle (if shape_type is 'circle')
        naca_code : str
            NACA 4-digit code for airfoil (if shape_type is 'airfoil')
        angle_of_attack : float
            Angle of attack in radians
        flow_speed : float
            Free stream flow speed
        """
        self.shape_type = shape_type
        self.radius = radius
        self.naca_code = naca_code
        self.angle_of_attack = angle_of_attack
        self.flow_speed = flow_speed
        
        # Grid parameters
        self.x_min, self.x_max = -5, 5
        self.y_min, self.y_max = -5, 5
        self.nx, self.ny = 100, 100
        
    def complex_potential(self, z):
        """
        Compute the complex potential at point z.
        
        Parameters:
        -----------
        z : complex
            Point in the complex plane
        
        Returns:
        --------
        complex
            Complex potential at point z
        """
        if self.shape_type == "circle":
            # Flow around a circle
            U = self.flow_speed
            R = self.radius
            alpha = self.angle_of_attack
            
            # Complex potential for flow around a circle
            w = U * (z + R**2/z) * np.exp(-1j*alpha)
            return w
        
        elif self.shape_type == "airfoil":
            # For airfoil, we use the Joukowski transformation
            # This is a simplified implementation
            c = 1.0  # Scale parameter
            
            # Map z to zeta (inverse Joukowski)
            # This is an approximation
            zeta = 0.5 * (z + np.sqrt(z**2 - 4*c**2))
            
            # Complex potential in the zeta-plane (circle)
            U = self.flow_speed
            alpha = self.angle_of_attack
            R = np.abs(zeta)
            
            w = U * (zeta + c**2/zeta) * np.exp(-1j*alpha)
            return w
        
        else:
            # Default to uniform flow
            return self.flow_speed * z * np.exp(-1j*self.angle_of_attack)
    
    def compute_velocity(self, z):
        """
        Compute the velocity at point z.
        
        Parameters:
        -----------
        z : complex
            Point in the complex plane
        
        Returns:
        --------
        complex
            Complex velocity at point z
        """
        # Velocity is the conjugate of the derivative of the complex potential
        h = 1e-6  # Small step for numerical differentiation
        dw = (self.complex_potential(z + h) - self.complex_potential(z)) / h
        return np.conj(dw)
    
    def compute_streamlines(self):
        """
        Compute streamlines for visualization.
        
        Returns:
        --------
        tuple
            (X, Y, Psi) where X, Y are meshgrid coordinates and Psi is the stream function
        """
        x = np.linspace(self.x_min, self.x_max, self.nx)
        y = np.linspace(self.y_min, self.y_max, self.ny)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j*Y
        
        # Compute complex potential
        W = np.zeros_like(Z, dtype=complex)
        for i in range(self.nx):
            for j in range(self.ny):
                if self.shape_type == "circle" and np.abs(Z[j, i]) <= self.radius:
                    # Inside the circle
                    W[j, i] = 0
                else:
                    W[j, i] = self.complex_potential(Z[j, i])
        
        # Stream function is the imaginary part of the complex potential
        Psi = np.imag(W)
        return X, Y, Psi
    
    def compute_velocity_field(self):
        """
        Compute velocity field for visualization.
        
        Returns:
        --------
        tuple
            (X, Y, U, V) where X, Y are meshgrid coordinates and U, V are velocity components
        """
        x = np.linspace(self.x_min, self.x_max, self.nx)
        y = np.linspace(self.y_min, self.y_max, self.ny)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j*Y
        
        # Compute velocity
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(self.nx):
            for j in range(self.ny):
                if self.shape_type == "circle" and np.abs(Z[j, i]) <= self.radius:
                    # Inside the circle
                    U[j, i] = V[j, i] = 0
                else:
                    vel = self.compute_velocity(Z[j, i])
                    U[j, i] = np.real(vel)
                    V[j, i] = np.imag(vel)
        
        return X, Y, U, V
    
    def visualize_flow(self):
        """
        Visualize the flow field.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        # Compute streamlines
        X, Y, Psi = self.compute_streamlines()
        
        # Compute velocity field
        _, _, U, V = self.compute_velocity_field()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot streamlines
        levels = np.linspace(np.min(Psi), np.max(Psi), 20)
        ax.contour(X, Y, Psi, levels=levels, colors='blue', alpha=0.7)
        
        # Plot velocity field (downsampled for clarity)
        skip = 5
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                 U[::skip, ::skip], V[::skip, ::skip], 
                 scale=50, width=0.002, color='red', alpha=0.7)
        
        # Plot the shape
        if self.shape_type == "circle":
            circle = Circle((0, 0), self.radius, fill=True, color='gray', alpha=0.5)
            ax.add_patch(circle)
        elif self.shape_type == "airfoil":
            # Generate NACA airfoil points
            # This is a simplified implementation for NACA 4-digit
            t = float(self.naca_code[2:]) / 100  # thickness
            
            # Generate x-coordinates
            x = np.linspace(0, 1, 100)
            
            # Compute thickness distribution
            yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
            
            # Rotate and scale
            x_rot = x * np.cos(self.angle_of_attack) - yt * np.sin(self.angle_of_attack)
            y_rot = x * np.sin(self.angle_of_attack) + yt * np.cos(self.angle_of_attack)
            
            # Plot upper surface
            ax.plot(x_rot, y_rot, 'k-', linewidth=2)
            # Plot lower surface
            ax.plot(x_rot, -y_rot, 'k-', linewidth=2)
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Flow around a {self.shape_type}, angle of attack = {np.degrees(self.angle_of_attack):.1f}°')
        ax.grid(True)
        
        return fig
    
    def compute_taylor_series(self, z0, order=5):
        """
        Compute the Taylor series expansion of the complex potential around z0.
        
        Parameters:
        -----------
        z0 : complex
            Point around which to expand
        order : int
            Order of the Taylor series
        
        Returns:
        --------
        list
            Coefficients of the Taylor series
        """
        # Use sympy for symbolic differentiation
        z = sp.Symbol('z', complex=True)
        
        if self.shape_type == "circle":
            U = self.flow_speed
            R = self.radius
            alpha = self.angle_of_attack
            
            # Complex potential for flow around a circle
            w_expr = U * (z + R**2/z) * sp.exp(-1j*alpha)
            
            # Compute Taylor series
            taylor_series = []
            for n in range(order + 1):
                if n == 0:
                    coeff = complex(w_expr.subs(z, z0))
                else:
                    # Compute nth derivative
                    w_diff = sp.diff(w_expr, z, n)
                    coeff = complex(w_diff.subs(z, z0)) / sp.factorial(n)
                
                taylor_series.append(coeff)
            
            return taylor_series
        
        else:
            # For other shapes, use numerical differentiation
            # This is a simplified implementation
            taylor_series = [self.complex_potential(z0)]
            
            for n in range(1, order + 1):
                # Compute nth derivative using finite differences
                h = 1e-4
                derivative = 0
                
                for k in range(n + 1):
                    binomial = sp.binomial(n, k)
                    derivative += (-1)**(n-k) * binomial * self.complex_potential(z0 + k*h)
                
                derivative /= h**n
                taylor_series.append(derivative / np.math.factorial(n))
            
            return taylor_series
    
    def compute_laurent_series(self, z0, inner_order=5, outer_order=5):
        """
        Compute the Laurent series expansion of the complex potential around z0.
        
        Parameters:
        -----------
        z0 : complex
            Point around which to expand
        inner_order : int
            Order of the inner part of the Laurent series (negative powers)
        outer_order : int
            Order of the outer part of the Laurent series (non-negative powers)
        
        Returns:
        --------
        list
            Coefficients of the Laurent series, indexed from -inner_order to outer_order
        """
        # For circle, we can compute Laurent series analytically
        if self.shape_type == "circle":
            U = self.flow_speed
            R = self.radius
            alpha = self.angle_of_attack
            
            # Initialize coefficients
            laurent_series = [0] * (inner_order + outer_order + 1)
            
            # Set the coefficients based on the complex potential formula
            # For circle: w(z) = U * (z + R^2/z) * exp(-i*alpha)
            
            # Term for z
            laurent_series[inner_order + 1] = U * np.exp(-1j*alpha)
            
            # Term for 1/z
            laurent_series[inner_order - 1] = U * R**2 * np.exp(-1j*alpha)
            
            return laurent_series
        
        else:
            # For other shapes, use numerical methods
            # This is a simplified implementation
            laurent_series = [0] * (inner_order + outer_order + 1)
            
            # Use contour integration to compute coefficients
            for n in range(-inner_order, outer_order + 1):
                # Compute contour integral
                num_points = 100
                theta = np.linspace(0, 2*np.pi, num_points)
                r = 2.0  # Radius of contour
                
                integral = 0
                for t in theta:
                    z = z0 + r * np.exp(1j*t)
                    integral += self.complex_potential(z) * np.exp(-1j*n*t) * 1j * r * np.exp(1j*t)
                
                integral *= 1 / (2*np.pi*1j) * (2*np.pi / num_points)
                laurent_series[inner_order + n] = integral
            
            return laurent_series