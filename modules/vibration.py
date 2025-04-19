import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

class VibrationAnalyzer:
    """
    Class for analyzing vibration modes of various structures.
    """
    
    def __init__(self, structure_type="mass_spring", n_masses=3, spring_constant=10.0, 
                 mass_value=1.0, beam_length=1.0, beam_elements=10):
        """
        Initialize the vibration analyzer.
        
        Parameters:
        -----------
        structure_type : str
            Type of structure to analyze ('mass_spring', 'beam', 'plate', 'custom')
        n_masses : int
            Number of masses in the mass-spring system
        spring_constant : float
            Spring constant (N/m)
        mass_value : float
            Mass value (kg)
        beam_length : float
            Length of the beam (m)
        beam_elements : int
            Number of elements for beam discretization
        """
        self.structure_type = structure_type
        self.n_masses = n_masses
        self.spring_constant = spring_constant
        self.mass_value = mass_value
        self.beam_length = beam_length
        self.beam_elements = beam_elements
        
        # Initialize matrices
        self.M = None  # Mass matrix
        self.K = None  # Stiffness matrix
        
        # Initialize results
        self.frequencies = None
        self.modes = None
        
        # Build matrices based on structure type
        self._build_matrices()
    
    def _build_matrices(self):
        """
        Build mass and stiffness matrices based on the structure type.
        """
        if self.structure_type == "mass_spring":
            # Mass-spring system
            # Mass matrix (diagonal)
            self.M = np.eye(self.n_masses) * self.mass_value
            
            # Stiffness matrix (tridiagonal)
            self.K = np.zeros((self.n_masses, self.n_masses))
            
            # Fill diagonal
            for i in range(self.n_masses):
                if i == 0:
                    # First mass connected to wall and next mass
                    self.K[i, i] = 2 * self.spring_constant
                elif i == self.n_masses - 1:
                    # Last mass connected to previous mass and wall
                    self.K[i, i] = 2 * self.spring_constant
                else:
                    # Middle masses connected to two adjacent masses
                    self.K[i, i] = 2 * self.spring_constant
            
            # Fill off-diagonal
            for i in range(self.n_masses - 1):
                self.K[i, i+1] = -self.spring_constant
                self.K[i+1, i] = -self.spring_constant
        
        elif self.structure_type == "beam":
            # Beam using finite element method
            # This is a simplified implementation for a beam with fixed-free boundary conditions
            
            # Number of degrees of freedom (2 per node: displacement and rotation)
            ndof = 2 * (self.beam_elements + 1)
            
            # Element length
            L = self.beam_length / self.beam_elements
            
            # Beam properties
            E = 2.1e11  # Young's modulus (Pa)
            I = 1e-6    # Area moment of inertia (m^4)
            rho = 7800  # Density (kg/m^3)
            A = 1e-4    # Cross-sectional area (m^2)
            
            # Initialize matrices
            self.M = np.zeros((ndof, ndof))
            self.K = np.zeros((ndof, ndof))
            
            # Element mass and stiffness matrices
            Me = np.array([
                [156, 22*L, 54, -13*L],
                [22*L, 4*L**2, 13*L, -3*L**2],
                [54, 13*L, 156, -22*L],
                [-13*L, -3*L**2, -22*L, 4*L**2]
            ]) * rho * A * L / 420
            
            Ke = np.array([
                [12, 6*L, -12, 6*L],
                [6*L, 4*L**2, -6*L, 2*L**2],
                [-12, -6*L, 12, -6*L],
                [6*L, 2*L**2, -6*L, 4*L**2]
            ]) * E * I / L**3
            
            # Assemble global matrices
            for e in range(self.beam_elements):
                # Global indices
                i1 = 2 * e
                i2 = 2 * e + 1
                i3 = 2 * (e + 1)
                i4 = 2 * (e + 1) + 1
                
                # Add element contributions to global matrices
                idx = [i1, i2, i3, i4]
                for i in range(4):
                    for j in range(4):
                        self.M[idx[i], idx[j]] += Me[i, j]
                        self.K[idx[i], idx[j]] += Ke[i, j]
            
            # Apply boundary conditions (fixed at left end)
            # Remove first two DOFs (displacement and rotation at left end)
            self.M = self.M[2:, 2:]
            self.K = self.K[2:, 2:]
        
        elif self.structure_type == "plate":
            # Simplified plate model using finite element method
            # This is a placeholder implementation
            # In a real application, this would be more complex
            
            # For simplicity, we'll create a small plate model
            nx, ny = 5, 5  # Number of nodes in x and y directions
            ndof = nx * ny  # Total degrees of freedom
            
            # Initialize matrices
            self.M = np.eye(ndof) * self.mass_value
            self.K = np.zeros((ndof, ndof))
            
            # Simple stiffness pattern for demonstration
            for i in range(ndof):
                # Diagonal terms
                self.K[i, i] = 4 * self.spring_constant
                
                # Connect to neighbors
                if i % nx > 0:  # Left neighbor
                    self.K[i, i-1] = -self.spring_constant
                    self.K[i-1, i] = -self.spring_constant
                
                if i % nx < nx - 1:  # Right neighbor
                    self.K[i, i+1] = -self.spring_constant
                    self.K[i+1, i] = -self.spring_constant
                
                if i >= nx:  # Bottom neighbor
                    self.K[i, i-nx] = -self.spring_constant
                    self.K[i-nx, i] = -self.spring_constant
                
                if i < ndof - nx:  # Top neighbor
                    self.K[i, i+nx] = -self.spring_constant
                    self.K[i+nx, i] = -self.spring_constant
        
        else:  # Custom structure
            # For custom structures, matrices should be provided externally
            # Here we just initialize with default values
            self.M = np.eye(self.n_masses) * self.mass_value
            self.K = np.eye(self.n_masses) * self.spring_constant
    
    def compute_modes(self):
        """
        Compute natural frequencies and mode shapes.
        
        Returns:
        --------
        tuple
            (frequencies, modes) where frequencies are in Hz and modes are the eigenvectors
        """
        # Solve the generalized eigenvalue problem: K*x = lambda*M*x
        eigenvalues, eigenvectors = la.eigh(self.K, self.M)
        
        # Natural frequencies (in Hz)
        self.frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
        
        # Mode shapes
        self.modes = eigenvectors
        
        return self.frequencies, self.modes
    
    def visualize_mode(self, mode_index):
        """
        Visualize a specific vibration mode.
        
        Parameters:
        -----------
        mode_index : int
            Index of the mode to visualize
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        if self.frequencies is None or self.modes is None:
            self.compute_modes()
        
        if mode_index < 0 or mode_index >= len(self.frequencies):
            raise ValueError(f"Mode index out of range. Should be between 0 and {len(self.frequencies)-1}")
        
        # Get the mode shape
        mode_shape = self.modes[:, mode_index]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.structure_type == "mass_spring":
            # For mass-spring system, plot the displacement of each mass
            x = np.arange(self.n_masses)
            
            # Normalize mode shape for visualization
            mode_shape = mode_shape / np.max(np.abs(mode_shape))
            
            # Plot equilibrium position
            ax.plot(x, np.zeros_like(x), 'ko', markersize=10, label='Equilibrium')
            
            # Plot mode shape
            ax.plot(x, mode_shape, 'ro-', markersize=8, label=f'Mode {mode_index+1}')
            
            # Add springs
            for i in range(self.n_masses):
                # Spring to the left (wall or previous mass)
                if i == 0:
                    # First mass connected to wall
                    ax.plot([-.5, 0], [0, 0], 'k-', linewidth=1)
                    ax.plot([-.5, 0], [0, mode_shape[i]], 'g--', linewidth=1)
                else:
                    # Connected to previous mass
                    ax.plot([i-1, i], [0, 0], 'k-', linewidth=1)
                    ax.plot([i-1, i], [mode_shape[i-1], mode_shape[i]], 'g--', linewidth=1)
                
                # Spring to the right (wall or next mass)
                if i == self.n_masses - 1:
                    # Last mass connected to wall
                    ax.plot([i, i+.5], [0, 0], 'k-', linewidth=1)
                    ax.plot([i, i+.5], [mode_shape[i], 0], 'g--', linewidth=1)
            
            ax.set_xlabel('Mass Index')
            ax.set_ylabel('Displacement')
            ax.set_title(f'Mode {mode_index+1}: {self.frequencies[mode_index]:.2f} Hz')
            ax.grid(True)
            ax.legend()
            
        elif self.structure_type == "beam":
            # For beam, we need to reconstruct the full mode shape
            # Each node has 2 DOFs: displacement and rotation
            
            # Number of elements
            n_elements = (len(mode_shape) + 2) // 2 - 1
            
            # Create x coordinates
            x = np.linspace(0, self.beam_length, n_elements + 1)
            
            # Extract displacements (every other DOF)
            displacements = np.zeros(n_elements + 1)
            displacements[1:] = mode_shape[::2]  # Skip the fixed end
            
            # Normalize for visualization
            displacements = displacements / np.max(np.abs(displacements)) * 0.2 * self.beam_length
            
            # Plot beam at equilibrium
            ax.plot(x, np.zeros_like(x), 'k-', linewidth=2, label='Equilibrium')
            
            # Plot mode shape
            ax.plot(x, displacements, 'r-', linewidth=2, label=f'Mode {mode_index+1}')
            
            ax.set_xlabel('Position along beam (m)')
            ax.set_ylabel('Displacement')
            ax.set_title(f'Beam Mode {mode_index+1}: {self.frequencies[mode_index]:.2f} Hz')
            ax.grid(True)
            ax.legend()
            
        elif self.structure_type == "plate":
            # For plate, we need to reshape the mode shape into a 2D grid
            nx, ny = 5, 5  # Number of nodes in x and y directions (from _build_matrices)
            
            # Reshape mode shape into 2D grid
            mode_2d = mode_shape.reshape((ny, nx))
            
            # Normalize for visualization
            mode_2d = mode_2d / np.max(np.abs(mode_2d))
            
            # Create coordinate grids
            x = np.linspace(0, 1, nx)
            y = np.linspace(0, 1, ny)
            X, Y = np.meshgrid(x, y)
            
            # Plot as surface
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, mode_2d, cmap='viridis', edgecolor='none')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Displacement')
            ax.set_title(f'Plate Mode {mode_index+1}: {self.frequencies[mode_index]:.2f} Hz')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        return fig
    
    def get_mass_matrix_latex(self):
        """
        Get LaTeX representation of the mass matrix.
        
        Returns:
        --------
        str
            LaTeX representation of the mass matrix
        """
        if self.M is None:
            return r"M \text{ not initialized}"
        
        # For large matrices, show a simplified representation
        if self.M.shape[0] > 5:
            return r"M = \begin{bmatrix} m & 0 & \cdots & 0 \\ 0 & m & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & m \end{bmatrix}"
        
        # For small matrices, show the actual values
        latex = r"M = \begin{bmatrix} "
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                latex += f"{self.M[i, j]:.1f}"
                if j < self.M.shape[1] - 1:
                    latex += " & "
            if i < self.M.shape[0] - 1:
                latex += r" \\ "
        latex += r" \end{bmatrix}"
        
        return latex
    
    def get_stiffness_matrix_latex(self):
        """
        Get LaTeX representation of the stiffness matrix.
        
        Returns:
        --------
        str
            LaTeX representation of the stiffness matrix
        """
        if self.K is None:
            return r"K \text{ not initialized}"
        
        # For large matrices, show a simplified representation
        if self.K.shape[0] > 5:
            return r"K = \begin{bmatrix} 2k & -k & 0 & \cdots & 0 \\ -k & 2k & -k & \cdots & 0 \\ 0 & -k & 2k & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 2k \end{bmatrix}"
        
        # For small matrices, show the actual values
        latex = r"K = \begin{bmatrix} "
        for i in range(self.K.shape[0]):
            for j in range(self.K.shape[1]):
                latex += f"{self.K[i, j]:.1f}"
                if j < self.K.shape[1] - 1:
                    latex += " & "
            if i < self.K.shape[0] - 1:
                latex += r" \\ "
        latex += r" \end{bmatrix}"
        
        return latex