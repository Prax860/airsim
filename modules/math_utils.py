import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def verify_cauchy_riemann(f, z0, h=1e-6):
    """
    Verify the Cauchy-Riemann equations for a complex function at a point.
    
    Parameters:
    -----------
    f : function
        Complex function f(z)
    z0 : complex
        Point at which to verify the equations
    h : float
        Step size for numerical differentiation
    
    Returns:
    --------
    tuple
        (is_satisfied, u_x, v_y, u_y, v_x) where is_satisfied is a boolean
        and the others are the partial derivatives
    """
    # Compute partial derivatives numerically
    f_z0 = f(z0)
    f_z0_dx = f(z0 + h)
    f_z0_dy = f(z0 + 1j*h)
    
    # Extract real and imaginary parts
    u_z0 = np.real(f_z0)
    v_z0 = np.imag(f_z0)
    
    u_z0_dx = np.real(f_z0_dx)
    v_z0_dx = np.imag(f_z0_dx)
    
    u_z0_dy = np.real(f_z0_dy)
    v_z0_dy = np.imag(f_z0_dy)
    
    # Compute partial derivatives
    u_x = (u_z0_dx - u_z0) / h
    v_x = (v_z0_dx - v_z0) / h
    
    u_y = (u_z0_dy - u_z0) / (1j*h)
    v_y = (v_z0_dy - v_z0) / (1j*h)
    
    # Check if Cauchy-Riemann equations are satisfied
    tol = 1e-6
    is_satisfied = (abs(u_x - v_y) < tol) and (abs(u_y + v_x) < tol)
    
    return is_satisfied, u_x, v_y, u_y, v_x

def compute_taylor_series(f, z0, order=5):
    """
    Compute the Taylor series expansion of a function around a point.
    
    Parameters:
    -----------
    f : function
        Function to expand
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
    z = sp.Symbol('z')
    
    # Convert function to sympy expression
    # This is a simplified approach - in a real application, you would need to handle different functions
    try:
        f_expr = f(z)
    except:
        # If f cannot be called with a symbolic argument, use numerical differentiation
        coeffs = [f(z0)]
        
        for n in range(1, order + 1):
            # Compute nth derivative using finite differences
            h = 1e-4
            derivative = 0
            
            for k in range(n + 1):
                binomial = sp.binomial(n, k)
                derivative += (-1)**(n-k) * binomial * f(z0 + k*h)
            
            derivative /= h**n
            coeffs.append(derivative / np.math.factorial(n))
        
        return coeffs
    
    # Continuing from where we left off in the compute_taylor_series function
    
    # Compute Taylor series
    taylor_series = []
    for n in range(order + 1):
        if n == 0:
            coeff = complex(f_expr.subs(z, z0))
        else:
            # Compute nth derivative
            f_diff = sp.diff(f_expr, z, n)
            coeff = complex(f_diff.subs(z, z0)) / sp.factorial(n)
        
        taylor_series.append(coeff)
    
    return taylor_series

def compute_laurent_series(f, z0, inner_order=5, outer_order=5):
    """
    Compute the Laurent series expansion of a function around a point.
    
    Parameters:
    -----------
    f : function
        Function to expand
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
    # Initialize coefficients
    laurent_series = [0] * (inner_order + outer_order + 1)
    
    # Use contour integration to compute coefficients
    for n in range(-inner_order, outer_order + 1):
        # Compute contour integral
        num_points = 100
        theta = np.linspace(0, 2*np.pi, num_points)
        r = 0.5  # Radius of contour
        
        integral = 0
        for t in theta:
            z = z0 + r * np.exp(1j*t)
            integral += f(z) * np.exp(-1j*n*t) * 1j * r * np.exp(1j*t)
        
        integral *= 1 / (2*np.pi*1j) * (2*np.pi / num_points)
        laurent_series[inner_order + n] = integral
    
    return laurent_series

def plot_complex_function(f, x_range=(-2, 2), y_range=(-2, 2), resolution=100, component='abs'):
    """
    Plot a complex function.
    
    Parameters:
    -----------
    f : function
        Complex function f(z)
    x_range : tuple
        Range of x values (real part)
    y_range : tuple
        Range of y values (imaginary part)
    resolution : int
        Resolution of the plot
    component : str
        Component to plot ('abs', 'real', 'imag', 'phase')
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    # Compute function values
    W = np.zeros_like(Z, dtype=complex)
    for i in range(resolution):
        for j in range(resolution):
            try:
                W[i, j] = f(Z[i, j])
            except:
                W[i, j] = np.nan
    
    # Extract component to plot
    if component == 'abs':
        values = np.abs(W)
        title = '|f(z)|'
    elif component == 'real':
        values = np.real(W)
        title = 'Re(f(z))'
    elif component == 'imag':
        values = np.imag(W)
        title = 'Im(f(z))'
    elif component == 'phase':
        values = np.angle(W)
        title = 'Arg(f(z))'
    else:
        values = np.abs(W)
        title = '|f(z)|'
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, values, cmap=cm.viridis, linewidth=0, antialiased=True)
    
    # Set labels and title
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_zlabel(title)
    ax.set_title(f'Complex Function Visualization: {title}')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig

def plot_convergence_region(f, z0, radius=2.0, resolution=100):
    """
    Plot the convergence region of a Taylor series.
    
    Parameters:
    -----------
    f : function
        Complex function f(z)
    z0 : complex
        Center of the Taylor series
    radius : float
        Radius of the plot
    resolution : int
        Resolution of the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    # Create grid
    theta = np.linspace(0, 2*np.pi, resolution)
    r = np.linspace(0, radius, resolution)
    Theta, R = np.meshgrid(theta, r)
    
    # Convert to Cartesian coordinates
    X = z0.real + R * np.cos(Theta)
    Y = z0.imag + R * np.sin(Theta)
    Z = X + 1j*Y
    
    # Compute function values
    W = np.zeros_like(Z, dtype=complex)
    for i in range(resolution):
        for j in range(resolution):
            try:
                W[i, j] = f(Z[i, j])
            except:
                W[i, j] = np.nan
    
    # Compute convergence measure (e.g., derivative magnitude)
    h = 1e-6
    dW = np.zeros_like(Z, dtype=float)
    for i in range(resolution):
        for j in range(resolution):
            try:
                dz = h * np.exp(1j*Theta[i, j])
                df = (f(Z[i, j] + dz) - f(Z[i, j])) / dz
                dW[i, j] = np.abs(df)
            except:
                dW[i, j] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot convergence measure
    im = ax.pcolormesh(X, Y, np.log10(dW), cmap='viridis', shading='auto')
    
    # Mark singularities
    # This is a simplified approach - in a real application, you would need to detect singularities
    ax.plot(z0.real, z0.imag, 'ro', markersize=10, label='Center')
    
    # Set labels and title
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title('Convergence Region Visualization')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log10(|f\'(z)|)')
    
    return fig

def compute_residue(f, z0, radius=1e-3, num_points=1000):
    """
    Compute the residue of a function at a point.
    
    Parameters:
    -----------
    f : function
        Complex function f(z)
    z0 : complex
        Point at which to compute the residue
    radius : float
        Radius of the contour
    num_points : int
        Number of points for numerical integration
    
    Returns:
    --------
    complex
        Residue of f at z0
    """
    # Compute contour integral
    theta = np.linspace(0, 2*np.pi, num_points)
    
    integral = 0
    for t in theta:
        z = z0 + radius * np.exp(1j*t)
        integral += f(z) * 1j * radius * np.exp(1j*t)
    
    integral *= 1 / (2*np.pi*1j) * (2*np.pi / num_points)
    
    return integral

def solve_eigenvalue_problem(A, B=None, n_eigvals=None):
    """
    Solve the generalized eigenvalue problem A*x = lambda*B*x.
    
    Parameters:
    -----------
    A : array_like
        Matrix A
    B : array_like, optional
        Matrix B. If None, solve the standard eigenvalue problem A*x = lambda*x
    n_eigvals : int, optional
        Number of eigenvalues to compute. If None, compute all eigenvalues
    
    Returns:
    --------
    tuple
        (eigenvalues, eigenvectors)
    """
    if B is None:
        # Standard eigenvalue problem
        if n_eigvals is None:
            eigenvalues, eigenvectors = np.linalg.eig(A)
        else:
            # Use scipy.linalg.eigs for partial eigendecomposition
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(A, eigvals=(0, min(n_eigvals, A.shape[0])-1))
    else:
        # Generalized eigenvalue problem
        if n_eigvals is None:
            eigenvalues, eigenvectors = np.linalg.eig(A, B)
        else:
            # Use scipy.linalg.eigs for partial eigendecomposition
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(A, B, eigvals=(0, min(n_eigvals, A.shape[0])-1))
    
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

def plot_matrix(A, title='Matrix Visualization'):
    """
    Plot a matrix as a heatmap.
    
    Parameters:
    -----------
    A : array_like
        Matrix to plot
    title : str
        Title of the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot matrix
    im = ax.imshow(A, cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Value')
    
    # Set labels and title
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title)
    
    # Add grid
    ax.grid(False)
    
    # Add values
    if A.shape[0] <= 10 and A.shape[1] <= 10:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f'{A[i, j]:.2f}', ha='center', va='center', color='white')
    
    return fig