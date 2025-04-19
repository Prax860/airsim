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
page = st.sidebar.radio('Go to', ['Home', 'Fluid Flow', 'Vibration Analysis', 'Signal Analysis'])

# Update session state
st.session_state.page = page

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
        
        # if st.button('Go to Fluid Flow'):
        #     st.session_state.page = 'Fluid Flow'
        #     st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
        
        create_animated_card(
            "📊 Signal Analysis", 
            "Analyze signals, detect faults, and explore frequency domain representations.",
            "📊"
        )
        
        # if st.button('Go to Signal Analysis'):
        #     st.session_state.page = 'Signal Analysis'
        #     st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    
    with col2:
        create_animated_card(
            "📳 Vibration Analysis", 
            "Analyze vibration modes of structures like beams, plates, and mass-spring systems.",
            "📳"
        )
        
        # if st.button('Go to Vibration Analysis'):
        #     st.session_state.page = 'Vibration Analysis'
        #     st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
        
        create_animated_card(
            "🧮 Math Utilities", 
            "Explore complex mathematical concepts like Laurent series, Taylor series, and more.",
            "🧮"
        )
        
        # if st.button('Go to Math Utilities'):
        #     st.session_state.page = 'Math Utilities'
        #     st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    
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
    st.title('🌊 Fluid Flow Simulation')
    
    # Add animation
    with st.spinner("Loading fluid flow simulator..."):
        display_lottie_animation(fluid_lottie, key="fluid_main", height=200)
    
    # Create tabs with animation
    tabs = create_tabs_with_animation(["Configuration", "Simulation", "Results", "Theory"])
    
    with tabs[0]:
        st.header("Configure Simulation")
        st.header("Shape Selection")
        shape_type = st.selectbox("Select shape", ["Circle", "Airfoil (NACA 4-digit)", "Custom"])
        
        if shape_type == "Circle":
            radius = st.slider("Radius", 0.1, 2.0, 1.0, 0.1)
            angle = st.slider("Angle of attack (degrees)", -15.0, 15.0, 0.0, 0.5)
            flow_speed = st.slider("Flow speed", 1.0, 50.0, 10.0, 0.5)
            
            # Initialize simulator
            simulator = FluidFlowSimulator(shape_type="circle", radius=radius, 
                                          angle_of_attack=np.radians(angle), 
                                          flow_speed=flow_speed)
            
            # Run simulation
            fig = simulator.visualize_flow()
            st.pyplot(fig)
            
            if show_math:
                st.subheader("Mathematical Details")
                st.latex(r"\Phi(z) = U_\infty \left(z + \frac{R^2}{z}\right) \cos\alpha")
                st.latex(r"\Psi(z) = U_\infty \left(z - \frac{R^2}{z}\right) \sin\alpha")
                st.write("Complex potential around a circle:")
                st.latex(r"F(z) = \Phi + i\Psi = U_\infty \left(z + \frac{R^2}{z}\right)e^{-i\alpha}")
        
        elif shape_type == "Airfoil (NACA 4-digit)":
            st.write("NACA Airfoil Parameters")
            naca_code = st.text_input("NACA 4-digit code", "0012")
            angle = st.slider("Angle of attack (degrees)", -15.0, 15.0, 5.0, 0.5)
            flow_speed = st.slider("Flow speed", 1.0, 50.0, 10.0, 0.5)
            
            # Initialize simulator
            simulator = FluidFlowSimulator(shape_type="airfoil", naca_code=naca_code,
                                          angle_of_attack=np.radians(angle), 
                                          flow_speed=flow_speed)
            
            # Run simulation
            fig = simulator.visualize_flow()
            st.pyplot(fig)
            
            if show_math:
                st.subheader("Mathematical Details")
                st.write("Joukowski transformation for airfoil generation:")
                st.latex(r"z = \zeta + \frac{c^2}{\zeta}")
                st.write("Where ζ is a point on a circle in the complex plane and z is the corresponding point on the airfoil.")

elif st.session_state.page == 'Vibration Analysis':
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
            st.write("Mass and stiffness matrices:")
            st.latex(analyzer.get_mass_matrix_latex())
            st.latex(analyzer.get_stiffness_matrix_latex())
            st.write("Eigenvalue problem:")
            st.latex(r"K\phi = \omega^2 M\phi")
            st.write("Where K is the stiffness matrix, M is the mass matrix, ω are the natural frequencies, and φ are the mode shapes.")

elif st.session_state.page == 'Signal Analysis':
    st.header("Signal Fault Detection")
    
    st.subheader("Signal Generation")
    signal_type = st.selectbox("Select signal type", 
                              ["Sine Wave", "Square Wave", "Sawtooth", "Custom", "Upload"])
    
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
        
        # Initialize analyzer
        analyzer = SignalAnalyzer(signal_type=signal_type.lower().replace(" ", "_"),
                                 frequency=frequency,
                                 amplitude=amplitude,
                                 duration=duration,
                                 sampling_rate=sampling_rate,
                                 noise_level=noise_level,
                                 add_fault=add_fault,
                                 fault_type=fault_type.lower().replace(" ", "_") if fault_type else None,
                                 fault_time=fault_time)
        
        # Generate and analyze signal
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
                st.subheader("Mathematical Details")
                st.write("Discrete Fourier Transform:")
                st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}")
                st.write("Where x[n] is the signal and X[k] is its frequency spectrum.")

# In the Signal Analysis section
elif st.session_state.page == 'Signal Analysis':
    # Add tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Time Domain", "FFT", "Wavelet Transform", "Laurent Series"])
    
    with viz_tab1:
        st.pyplot(analyzer.visualize_signal())
    
    with viz_tab2:
        st.pyplot(analyzer.visualize_fft())
    
    with viz_tab3:
        with st.spinner("Computing wavelet transform..."):
            st.pyplot(analyzer.visualize_wavelet())
    
    with viz_tab4:
        with st.spinner("Computing Laurent series..."):
            st.pyplot(analyzer.visualize_laurent())

# Create necessary directories
import os
os.makedirs("c:\\Users\\cwc\\Desktop\\airsim\\modules", exist_ok=True)
os.makedirs("c:\\Users\\cwc\\Desktop\\airsim\\tests", exist_ok=True)
os.makedirs("c:\\Users\\cwc\\Desktop\\airsim\\data\\sample_structures", exist_ok=True)
os.makedirs("c:\\Users\\cwc\\Desktop\\airsim\\data\\sample_signals", exist_ok=True)

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #ddd;">
    <p>AeroSim - Developed with ❤️ for Engineering Analysis</p>
</div>
""", unsafe_allow_html=True)