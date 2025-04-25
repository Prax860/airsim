# File: app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import base64
import os

# Set page config - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AeroSim",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Now import other modules
from modules.fluid_flow import FluidFlowSimulator
from modules.vibration import VibrationAnalyzer
from modules.signal_analysis import SignalAnalyzer
from modules.math_utils import (verify_cauchy_riemann, compute_taylor_series,
                              compute_laurent_series, plot_complex_function,
                              plot_convergence_region)
from modules.ui_utils import (load_lottie_url, display_lottie_animation,
                            apply_custom_css, animated_progress,
                            create_animated_counter, create_animated_card,
                            create_tabs_with_animation)

# Load custom CSS
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Try to load the CSS file
css_path = Path(__file__).parent / "static" / "style.css"
if css_path.exists():
    load_css(css_path)

# Apply custom CSS
apply_custom_css()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Create sidebar navigation with animation
st.sidebar.title('✈️ AeroSim Navigation')

# Add animation to sidebar
fluid_lottie = load_lottie_url('https://assets5.lottiefiles.com/packages/lf20_jR229r.json')
vibration_lottie = load_lottie_url('https://assets2.lottiefiles.com/packages/lf20_kbfzivr8.json')
signal_lottie = load_lottie_url('https://assets9.lottiefiles.com/packages/lf20_kseho6rf.json')
math_lottie = load_lottie_url('https://assets3.lottiefiles.com/packages/lf20_ydo1amjm.json')

with st.sidebar:
    st.markdown("---")
    display_lottie_animation(fluid_lottie, key="fluid_sidebar", height=100)
    st.markdown("---")
    # Add the show_math checkbox
    show_math = st.checkbox("Show Mathematical Details", value=False)

# Navigation
page = st.sidebar.radio('Go to', ['Home', 'Fluid Flow', 'Vibration Analysis', 'Signal Analysis', 'Math Utilities'], 
                        index=0 if st.session_state.page not in ['Home', 'Fluid Flow', 'Vibration Analysis', 'Signal Analysis', 'Math Utilities'] 
                        else ['Home', 'Fluid Flow', 'Vibration Analysis', 'Signal Analysis', 'Math Utilities'].index(st.session_state.page))

# Update session state only if page changed
if page != st.session_state.page:
    st.session_state.page = page
    # Reset graph display state when changing pages
    st.session_state['show_sim_graph'] = False
    st.session_state['sim_graph_type'] = None
    st.session_state['fluid_flow_tab'] = 0 # Reset to config tab

# Display the selected page
if st.session_state.page == 'Home':
    st.title('✈️ AeroSim: Airflow & Vibration Analyzer')

    # Add a welcome animation
    welcome_lottie = load_lottie_url('https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        display_lottie_animation(welcome_lottie, key="welcome", height=300)

    st.markdown("""
    <h3 style='text-align: center; color: #1E88E5; animation: fadeIn 2s;'>
        Welcome to AeroSim, a comprehensive simulation and analysis tool for engineering systems.
    </h3>

    <style>
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Create animated cards for each module
    st.markdown("## 🚀 Explore Our Modules")

    col1, col2 = st.columns(2)

    with col1:
        create_animated_card(
            "🌊 Fluid Flow Simulation",
            "Simulate airflow around various shapes using conformal mapping and complex potential theory.",
            "🌊"
        )

        if st.button('Go to Fluid Flow'):
            st.session_state.page = 'Fluid Flow'
            st.rerun()

        create_animated_card(
            "📊 Signal Analysis",
            "Analyze signals, detect faults, and explore frequency domain representations.",
            "📊"
        )

        if st.button('Go to Signal Analysis'):
            st.session_state.page = 'Signal Analysis'
            st.rerun()

    with col2:
        create_animated_card(
            "📳 Vibration Analysis",
            "Analyze vibration modes of structures like beams, plates, and mass-spring systems.",
            "📳"
        )

        if st.button('Go to Vibration Analysis'):
            st.session_state.page = 'Vibration Analysis'
            st.rerun()

        create_animated_card(
            "🧮 Math Utilities",
            "Explore complex mathematical concepts like Laurent series, Taylor series, and more.",
            "🧮"
        )

        if st.button('Go to Math Utilities'):
            st.session_state.page = 'Math Utilities'
            st.rerun()

    # Add some statistics with animated counters
    st.markdown("## 📈 AeroSim in Numbers")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div style='text-align: center;'>Simulation Modules</div>", unsafe_allow_html=True)
        create_animated_counter(0, 4, suffix=" modules")

    with col2:
        st.markdown("<div style='text-align: center;'>Mathematical Functions</div>", unsafe_allow_html=True)
        create_animated_counter(0, 15, suffix="+ functions")

    with col3:
        st.markdown("<div style='text-align: center;'>Visualization Types</div>", unsafe_allow_html=True)
        create_animated_counter(0, 10, suffix="+ types")

elif st.session_state.page == 'Fluid Flow':
    if st.button("Home"):
        st.session_state.page = 'Home'
        st.rerun()

    st.title('🌊 Fluid Flow Simulation')

    # # Add animation
    # with st.spinner("Loading fluid flow simulator..."):
    #     display_lottie_animation(fluid_lottie, key="fluid_main", height=200)

    # Create tabs with animation
    tabs = create_tabs_with_animation(["Configuration", "Simulation", "Results", "Theory"])


    with tabs[0]: # Configuration Tab
        st.header("Configure Simulation")
        st.header("Shape Selection")
        shape_type = st.selectbox("Select shape", ["Circle", "Airfoil (NACA 4-digit)", "Custom"])

        if shape_type == "Circle":
            radius = st.slider("Radius", 0.1, 2.0, 1.0, 0.1)
            angle = st.slider("Angle of attack (degrees)", -15.0, 15.0, 0.0, 0.5)
            flow_speed = st.slider("Flow speed", 1.0, 50.0, 10.0, 0.5)

            # Store parameters in session state for simulation tab
            st.session_state['sim_shape_type'] = "circle"
            st.session_state['sim_radius'] = radius
            st.session_state['sim_angle_of_attack'] = np.radians(angle)
            st.session_state['sim_flow_speed'] = flow_speed

            if st.button("Show Circle Flow Graph"):
                st.session_state['show_sim_graph'] = True
                st.session_state['sim_graph_type'] = "circle"
                st.session_state['fluid_flow_tab'] = 1 # Set tab to Simulation
                st.rerun() # Trigger rerun


        elif shape_type == "Airfoil (NACA 4-digit)":
            st.write("NACA Airfoil Parameters")
            naca_code = st.text_input("NACA 4-digit code", "0012")
            angle = st.slider("Angle of attack (degrees)", -15.0, 15.0, 5.0, 0.5)
            flow_speed = st.slider("Flow speed", 1.0, 50.0, 10.0, 0.5)

            # Store parameters in session state for simulation tab
            st.session_state['sim_shape_type'] = "airfoil"
            st.session_state['sim_naca_code'] = naca_code
            st.session_state['sim_angle_of_attack'] = np.radians(angle)
            st.session_state['sim_flow_speed'] = flow_speed

            if st.button("Show Airfoil Flow Graph"):
                st.session_state['show_sim_graph'] = True
                st.session_state['sim_graph_type'] = "airfoil"
                st.session_state['fluid_flow_tab'] = 1 # Set tab to Simulation
                st.rerun() # Trigger rerun


    with tabs[1]: # Simulation Tab
        st.header("Simulation")
        fullscreen = st.checkbox("Fullscreen graph", value=True)

        # Check session state to see if a graph should be shown
        if st.session_state.get('show_sim_graph', False):
            sim_graph_type = st.session_state.get('sim_graph_type', None)
            if sim_graph_type == "circle":
                # Retrieve parameters from session state
                radius = st.session_state.get('sim_radius', 1.0)
                angle_of_attack = st.session_state.get('sim_angle_of_attack', 0.0)
                flow_speed = st.session_state.get('sim_flow_speed', 10.0)
                simulator = FluidFlowSimulator(
                    shape_type="circle",
                    radius=radius,
                    angle_of_attack=angle_of_attack,
                    flow_speed=flow_speed
                )
                # Call the visualize_flow method and display the plot
                fig = simulator.visualize_flow()
                st.pyplot(fig)

            elif sim_graph_type == "airfoil":
                # Retrieve parameters from session state
                naca_code = st.session_state.get('sim_naca_code', "0012")
                angle_of_attack = st.session_state.get('sim_angle_of_attack', 0.0)
                flow_speed = st.session_state.get('sim_flow_speed', 10.0)
                simulator = FluidFlowSimulator(
                    shape_type="airfoil",
                    naca_code=naca_code,
                    angle_of_attack=angle_of_attack,
                    flow_speed=flow_speed
                )
                # Call the visualize_flow method and display the plot
                fig = simulator.visualize_flow()
                st.pyplot(fig)
        else:
            st.info("Configure simulation parameters in the 'Configuration' tab and click 'Show Graph'.")


    with tabs[2]: # Results Tab
        st.header("Results")
        st.write("Detailed simulation results and interpretation will be shown here.")
        st.info("(Placeholder for further analysis metrics like pressure coefficient, streamlines, etc.)")

        if show_math:
            st.subheader("Mathematical Details")
            # General mathematical details
            st.markdown("""
            *Complex Potential Flow Theory*: This fundamental concept in fluid dynamics combines the velocity potential (Φ) and the stream function (Ψ) into a single complex function, F(z) = Φ + iΨ, where z is a complex variable representing a point in the 2D flow field. For incompressible, irrotational flow, F(z) is an analytic function. The real part (Φ) gives the velocity potential, and its gradient gives the velocity components. The imaginary part (Ψ) gives the stream function, and streamlines are given by Ψ = constant. This theory simplifies the analysis of 2D potential flow problems by leveraging the powerful tools of complex analysis.

            *Bernoulli's Equation*: This principle relates the pressure, velocity, and elevation in a fluid flow. For steady, incompressible, inviscid flow along a streamline, Bernoulli's equation states that the total mechanical energy of the fluid remains constant. In the context of potential flow around an object, Bernoulli's equation is used to calculate the pressure distribution on the surface of the object based on the flow velocity derived from the complex potential. This pressure distribution is crucial for determining forces like lift and drag.

            """)

    with tabs[3]:  # Theory Tab
        st.header("Theory")
        st.write("This section explains the underlying theoretical concepts used in the Fluid Flow Simulation module.")

        st.markdown(r"""
        ## 🌊 Fluid Flow Simulation: Theoretical Foundations

        In our fluid flow simulation tool, users can select a shape (such as a circle or airfoil), and visualize the flow field and pressure distribution around it. This simulation is powered by classical fluid dynamics theory and elegant mathematical tools rooted in complex analysis.

        ### 🧠 Complex Potential Flow Theory

        This theory is a cornerstone for analyzing *2D, incompressible, irrotational fluid flow. It uses complex numbers to represent the flow using a single analytic function called the **complex potential*:

        $$F(z) = \phi(x, y) + i \psi(x, y)$$

        where:
        - \( F(z) \) is the complex potential,
        - \( \phi(x, y) \) is the *velocity potential*,
        - \( \psi(x, y) \) is the *stream function*,
        - \( z = x + iy \) is the complex coordinate.

        The flow velocity at any point is found by taking the *complex conjugate of the derivative* of the complex potential:

        $$v = \overline{F'(z)}$$

        This formulation solves Laplace’s equation for both the velocity potential and stream function, making it ideal for simulating ideal fluid flow.

        ### 🔁 Joukowski Transformation

        The *Joukowski Transformation* is a conformal mapping used to transform a circle into an airfoil-like shape:

        $$z = \zeta + \frac{a^2}{\zeta}$$

        where:
        - \( \zeta \): point in the circle (parameter) plane,
        - \( z \): mapped point in the physical plane,
        - \( a \): radius of the original circle.

        Because conformal mappings preserve angles, the irrotational flow characteristics remain unchanged. This allows us to:
        - Solve for flow around a circle,
        - Then map the solution to an airfoil shape.

        ### 💨 Bernoulli’s Equation

        After computing the velocity using the complex potential, we use *Bernoulli’s equation* to find pressure:

        $$P + \frac{1}{2} \rho v^2 = \text{constant}$$

        where:
        - \( P \): pressure,
        - \( \rho \): fluid density,
        - \( v \): flow speed (magnitude of velocity vector).

        This applies for *steady, incompressible, inviscid* flow along a streamline. Using \( v \) from the potential, we determine pressure variations around the shape, enabling computation of lift and drag.

        ### 🧪 What This Simulator Lets You Do

        - *Select a shape*: Circle, ellipse, or airfoil.
        - *Visualize streamlines*: From the imaginary part of the complex potential.
        - *Analyze flow behavior*: Velocity and pressure fields.
        - *Explore aerodynamics*: See how different shapes impact lift and drag forces.

        This tool brings classical fluid theory to life through an interactive, visual experience.
        """)

   


elif st.session_state.page == 'Vibration Analysis':
    if st.button("Home"):
        st.session_state.page = 'Home'
        st.rerun()

    st.title("📳 Vibration Analysis")

    st.header("Vibration Mode Solver")

    st.subheader("Structure Definition")
    structure_type = st.selectbox("Select structure type",
                                 ["Simple Mass-Spring System", "Beam", "Plate", "Custom"])

    if structure_type == "Simple Mass-Spring System":
        n_masses = st.slider("Number of masses", 2, 10, 3)
        spring_constant = st.slider("Spring constant (N/m)", 1.0, 100.0, 10.0)
        mass_value = st.slider("Mass value (kg)", 0.1, 10.0, 1.0)

        # Initialize analyzer
        analyzer = VibrationAnalyzer(structure_type="mass_spring",
                                    n_masses=n_masses,
                                    spring_constant=spring_constant,
                                    mass_value=mass_value)

        # Compute modes
        frequencies, modes = analyzer.compute_modes()

        # Display results
        st.subheader("Natural Frequencies")
        for i, freq in enumerate(frequencies):
            st.write(f"Mode {i+1}: {freq:.2f} Hz")

        # Visualize modes
        mode_to_view = st.slider("Select mode to visualize", 1, len(frequencies), 1)
        fig = analyzer.visualize_mode(mode_to_view-1)
        st.pyplot(fig)

        if show_math:
            st.subheader("Mathematical Details")

    st.markdown(r"""
    ## 🔩 Vibration Analysis Using Eigenvalues

    Vibration analysis is crucial in mechanical and structural systems to determine how a structure behaves when subjected to dynamic forces. In systems with multiple degrees of freedom (DOFs), such as a multi-mass-spring system or a beam, this behavior is modeled using *mass* and *stiffness matrices*.

    ### 🧱 Mass and Stiffness Matrices

    The system is described by:

    - *Mass Matrix \( M \)*: Represents the distribution of mass.
    - *Stiffness Matrix \( K \)*: Represents the stiffness and how elements are connected.

    For a system with \( n \) masses:
    - The *mass matrix* is usually diagonal.
    - The *stiffness matrix* is constructed from spring constants and their connections.

    *Mass matrix \( M \):*
    """)
    st.latex(analyzer.get_mass_matrix_latex())

    st.markdown("*Stiffness matrix \( K \):*")
    st.latex(analyzer.get_stiffness_matrix_latex())

    st.markdown(r"""
    ### 🧠 Eigenvalue Problem

    The system's free vibration behavior (without damping or external forces) is governed by the *generalized eigenvalue problem*:

    $$
    K\phi = \omega^2 M\phi
    $$

    Where:
    - \( K \) is the stiffness matrix  
    - \( M \) is the mass matrix  
    - \( \omega^2 \) is the eigenvalue  
    - \( \omega \) is the natural frequency  
    - \( \phi \) is the eigenvector (mode shape)

    This equation arises from substituting a harmonic solution \( x(t) = \phi \sin(\omega t) \) into the second-order differential equation of motion.

    ### 📉 What the Eigenvalues and Eigenvectors Mean

    - The *eigenvalues \( \omega^2 \)* give the *squares of the natural frequencies* of the structure.
    - Taking the square root gives the *natural frequencies \( \omega \)* (in radians/sec).
    - Each *eigenvector \( \phi \)* describes a *mode shape*, i.e., how the structure deforms when vibrating at that frequency.

    These are essential for:
    - Predicting *resonance* (when external forces match natural frequencies),
    - Designing to *avoid failure* due to vibrations,
    - Understanding *how and where* a structure will flex or move dynamically.

    ### 🧪 How This Simulator Uses CVLA and Eigenvalue Analysis

    Our simulation tool leverages *Complex-Valued Linear Algebra (CVLA)* techniques to:
    - Efficiently solve the eigenvalue problem \( K\phi = \omega^2 M\phi \)
    - Compute all natural frequencies and mode shapes
    - Handle arbitrary user-defined or system-generated matrices
    - Visualize dynamic motion corresponding to each mode

    These calculations form the foundation of the *vibration analysis tab* in your app. The results let users explore how a structure vibrates, which modes dominate, and how system parameters affect its behavior.
    """)

elif st.session_state.page == 'Signal Analysis':
    if st.button("Home"):
        st.session_state.page = 'Home'
        st.rerun()

    st.header("Signal Fault Detection")

    st.subheader("Signal Generation")
    signal_type = st.selectbox("Select signal type",
                              ["Sine Wave", "Square Wave", "Sawtooth", "Custom", "Upload"])

    # Initialize analyzer only if a signal type is selected for generation
    analyzer = None
    if signal_type in ["Sine Wave", "Square Wave", "Sawtooth"]:
        frequency = st.slider("Frequency (Hz)", 0.1, 10.0, 1.0, 0.1)
        amplitude = st.slider("Amplitude", 0.1, 10.0, 1.0, 0.1)
        duration = st.slider("Duration (s)", 1.0, 10.0, 5.0, 0.5)
        sampling_rate = st.slider("Sampling rate (Hz)", 10, 1000, 100)

        # Add noise and fault options
        add_noise = st.checkbox("Add noise", value=True)
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.1, 0.05) if add_noise else 0.0

        add_fault = st.checkbox("Add fault", value=True)
        fault_type = st.selectbox("Fault type", ["Impulse", "Step Change", "Frequency Shift"]) if add_fault else None
        fault_time = st.slider("Fault time (s)", 0.0, duration, duration/2, 0.1) if add_fault else None

        analyzer = SignalAnalyzer(signal_type=signal_type.lower().replace(" ", "_"),
                                 frequency=frequency,
                                 amplitude=amplitude,
                                 duration=duration,
                                 sampling_rate=sampling_rate,
                                 noise_level=noise_level,
                                 add_fault=add_fault,
                                 fault_type=fault_type.lower().replace(" ", "_") if fault_type else None,
                                 fault_time=fault_time)

        # Generate signal
        time, signal = analyzer.generate_signal()

        # Plot original signal
        st.subheader("Original Signal")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, signal)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        st.pyplot(fig)

        # Analyze signal
        st.subheader("Signal Analysis")
        analysis_type = st.selectbox("Analysis method",
                                    ["Fourier Transform", "Wavelet Transform", "Laurent Series"])

        if analysis_type == "Fourier Transform":
            freq, spectrum = analyzer.compute_fft()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(freq, np.abs(spectrum))
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.grid(True)
            st.pyplot(fig)

            # Fault detection
            detected_faults = analyzer.detect_faults_fft()
            if detected_faults:
                st.success("Faults detected!")
                for fault in detected_faults:
                    st.write(f"- {fault}")
            else:
                st.info("No faults detected.")
    if show_math:
        st.subheader("Mathematical Details: Fourier Transform")

        st.markdown(r"""
        ## 📡 Signal Fault Detection with Fourier Transform

        *Fourier Transform* is a mathematical technique that decomposes a time-based signal into its constituent *frequencies. This transformation helps in understanding which frequency components are present in the signal and their respective amplitudes. It's especially valuable in signal analysis where faults or anomalies might appear more clearly in the **frequency domain* than in the *time domain*.

        ### 🧮 Discrete Fourier Transform (DFT)

        For digital signals (sampled at discrete intervals), we use the *Discrete Fourier Transform (DFT). The efficient implementation of DFT is called the **Fast Fourier Transform (FFT)*.

        The DFT of a signal \( x[n] \) of length \( N \) is given by:

        $$ 
        X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi k n}{N}} 
        $$

        Where:
        - \( x[n] \): the time-domain signal
        - \( X[k] \): the frequency-domain representation
        - \( j \): the imaginary unit
        - \( N \): total number of samples

        ### 🔍 How It Helps in Fault Detection

        - *Normal signals* have predictable frequency patterns (e.g., a sine wave has one sharp frequency peak).
        - *Faulty signals* (e.g., with impulses or noise) introduce new, unexpected frequency components.
        - By analyzing \( X[k] \), you can detect these changes, making it easier to *diagnose issues* in real time.

        ### ⚠️ Practical Example:

        Suppose you generate a sine wave at 5 Hz:
        - Its frequency spectrum \( X[k] \) will show a *peak at 5 Hz*.

        If a fault (e.g., an impulse) is introduced:
        - The spectrum may now contain *high-frequency components* or *broadened peaks*, indicating an anomaly.

        ### ✅ Summary

        Fourier analysis:
        - Converts the signal from time to frequency domain
        - Helps highlight abnormalities not easily visible in the time domain
        - Enables automated fault detection in complex systems
        """)

        st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi k n}{N}}")

        st.write("Where:")
        st.markdown(r"""
        - \( x[n] \): signal in the time domain  
        - \( X[k] \): frequency-domain spectrum  
        - \( N \): number of samples  
        - \( k \): frequency bin  
        - \( j \): imaginary unit  
        """)


        # Add tabs for visualization only if analyzer is initialized
        if analyzer:
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Time Domain", "FFT", "Wavelet Transform", "Laurent Series"])

            with viz_tab1:
                st.pyplot(analyzer.visualize_signal())

            with viz_tab2:
                st.pyplot(analyzer.visualize_fft())

            # In the Signal Analysis section, update the visualization tabs
            
            with viz_tab3:
                with st.spinner("Computing wavelet transform..."):
                    try:
                        wavelet_fig = analyzer.visualize_wavelet()
                        st.pyplot(wavelet_fig)
                    except Exception as e:
                        st.error(f"Error computing wavelet transform: {str(e)}")
                        st.info("Try adjusting signal parameters or sampling rate for better wavelet analysis.")
                    if show_math:
                        st.subheader("Mathematical Details: Wavelet Transform")
                        st.markdown("""
                        *Wavelet Transform*: Unlike the Fourier Transform, which uses sine and cosine waves (basis functions localized in frequency but not time), the Wavelet Transform uses wavelets (basis functions localized in both time and frequency). This allows the Wavelet Transform to provide a time-frequency representation of a signal, making it particularly effective for analyzing non-stationary signals (signals whose frequency content changes over time) and detecting transient features like spikes or sudden changes, which are often indicative of faults.
                        """)
            
            
            with viz_tab4:
                with st.spinner("Computing Laurent series..."):
                    try:
                        laurent_fig = analyzer.visualize_laurent()
                        st.pyplot(laurent_fig)
                    except Exception as e:
                        st.error(f"Error computing Laurent series: {str(e)}")
                        st.info("Laurent series may not be applicable to all signal types. Try different signal parameters.")
                    if show_math:
                        st.subheader("Mathematical Details: Laurent Series")
                        st.markdown("""
                        *Laurent Series*: In complex analysis, a Laurent series is a representation of a complex function as a series of terms involving both positive and negative powers of \(z - c\). It is a generalization of the Taylor series and is used to analyze functions that have singularities (points where the function is not analytic). The Laurent series expansion around a singularity can reveal the nature of the singularity and the behavior of the function near that point. In signal processing, Laurent series can sometimes be applied in theoretical analysis or for specific types of signal representations, particularly when dealing with systems described by complex functions with poles.
                        """)
    else:
        st.info("Please select a signal type to generate and analyze.")


elif st.session_state.page == 'Math Utilities':
    if st.button("Home"):
        st.session_state.page = 'Home'
        st.rerun()

    st.title("🧮 Math Utilities")

    st.write("This section provides tools and visualizations for exploring various mathematical concepts relevant to engineering analysis, particularly in the domain of complex analysis.")

    st.subheader("Example: Verify Cauchy-Riemann Equations")
    st.write("The Cauchy-Riemann equations are a pair of partial differential equations that provide a necessary (and under certain conditions, sufficient) condition for a complex function \(f(z) = u(x, y) + i v(x, y)\) to be complex differentiable (analytic) at a point. Verifying these equations is a fundamental step in complex analysis to determine if a function behaves 'nicely' in the complex plane.")
    st.latex(r"\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}")
    st.latex(r"\frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}")
    st.write("If these equations hold for a function in a region, and the partial derivatives are continuous, then the function is analytic in that region.")
    # You would add input fields and call verify_cauchy_riemann from math_utils here

    st.subheader("Example: Compute Taylor Series")
    st.write("A Taylor series is a representation of a function as an infinite sum of terms that are calculated from the values of the function's derivatives at a single point. For functions that are analytic at a point, the Taylor series provides a local approximation of the function. It is a fundamental tool in calculus and analysis for approximating functions and evaluating integrals.")
    st.latex(r"f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \dots")
    st.latex(r"f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n")
    # You would add input fields and call compute_taylor_series from math_utils here

    st.subheader("Example: Compute Laurent Series")
    st.write("A Laurent series is a generalization of the Taylor series that allows for the representation of complex functions that have singularities. Unlike the Taylor series, which only involves non-negative powers, the Laurent series includes terms with negative powers. This makes it suitable for analyzing the behavior of a function around a point where it is not analytic.")
    st.latex(r"f(z) = \sum_{n=-\infty}^{\infty} a_n (z-c)^n")
    st.write("The part of the series with negative powers is called the principal part, and it provides information about the nature of the singularity at \(c\).")
    # You would add input fields and call compute_laurent_series from math_utils here

    st.subheader("Example: Plot Complex Functions and Convergence Regions")
    st.write("Visualizing complex functions and their regions of convergence is essential for understanding their behavior. Plotting the magnitude, phase, or real/imaginary parts of a complex function can reveal important features like poles, zeros, and branch cuts. Plotting convergence regions helps understand where series representations (like Taylor or Laurent series) are valid.")
    # You would add input fields and call plot_complex_function and plot_convergence_region from math_utils here


# Create necessary directories
user_base_path = os.path.expanduser("~")
base_dir = os.path.join(user_base_path, "airsim")

os.makedirs(os.path.join(base_dir, "modules"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "tests"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "data", "sample_structures"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "data", "sample_signals"), exist_ok=True)

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #ddd;">
    <p>AeroSim - Developed with ❤️ for Engineering Analysis</p>
</div>
""", unsafe_allow_html=True)