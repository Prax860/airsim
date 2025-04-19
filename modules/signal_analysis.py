import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sympy as sp

class SignalAnalyzer:
    """
    Class for analyzing signals and detecting faults using various methods.
    """
    
    def __init__(self, signal_type="sine_wave", frequency=1.0, amplitude=1.0, 
                 duration=5.0, sampling_rate=100, noise_level=0.1, 
                 add_fault=False, fault_type=None, fault_time=None):
        """
        Initialize the signal analyzer.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to generate ('sine_wave', 'square_wave', 'sawtooth', 'custom')
        frequency : float
            Frequency of the signal (Hz)
        amplitude : float
            Amplitude of the signal
        duration : float
            Duration of the signal (s)
        sampling_rate : int
            Sampling rate (Hz)
        noise_level : float
            Level of noise to add to the signal
        add_fault : bool
            Whether to add a fault to the signal
        fault_type : str
            Type of fault to add ('impulse', 'step_change', 'frequency_shift')
        fault_time : float
            Time at which to add the fault (s)
        """
        self.signal_type = signal_type
        self.frequency = frequency
        self.amplitude = amplitude
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        self.add_fault = add_fault
        self.fault_type = fault_type
        self.fault_time = fault_time
        
        # Initialize time and signal arrays
        self.time = None
        self.signal = None
        
        # Initialize analysis results
        self.fft_freq = None
        self.fft_spectrum = None
        self.wavelet_scales = None
        self.wavelet_coeffs = None
        self.laurent_coeffs = None
    
    def generate_signal(self):
        """
        Generate a signal based on the specified parameters.
        
        Returns:
        --------
        tuple
            (time, signal) arrays
        """
        # Create time array
        self.time = np.linspace(0, self.duration, int(self.duration * self.sampling_rate))
        
        # Generate base signal
        if self.signal_type == "sine_wave":
            self.signal = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time)
        
        elif self.signal_type == "square_wave":
            self.signal = self.amplitude * signal.square(2 * np.pi * self.frequency * self.time)
        
        elif self.signal_type == "sawtooth":
            self.signal = self.amplitude * signal.sawtooth(2 * np.pi * self.frequency * self.time)
        
        else:  # Custom or unknown type, default to sine
            self.signal = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time)
        
        # Add noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * self.amplitude, len(self.time))
            self.signal += noise
        
        # Add fault
        if self.add_fault and self.fault_type is not None and self.fault_time is not None:
            # Find index corresponding to fault time
            fault_idx = np.argmin(np.abs(self.time - self.fault_time))
            
            if self.fault_type == "impulse":
                # Add impulse fault
                self.signal[fault_idx] += 5 * self.amplitude
            
            elif self.fault_type == "step_change":
                # Add step change fault
                self.signal[fault_idx:] += 0.5 * self.amplitude
            
            elif self.fault_type == "frequency_shift":
                # Add frequency shift fault
                new_freq = 2 * self.frequency
                self.signal[fault_idx:] = self.amplitude * np.sin(2 * np.pi * new_freq * (self.time[fault_idx:] - self.time[fault_idx]) + 
                                                                 2 * np.pi * self.frequency * self.time[fault_idx])
        
        return self.time, self.signal
    
    def compute_fft(self):
        """
        Compute the Fast Fourier Transform of the signal.
        
        Returns:
        --------
        tuple
            (frequencies, spectrum) arrays
        """
        if self.signal is None:
            self.generate_signal()
        
        # Compute FFT
        n = len(self.signal)
        self.fft_spectrum = np.fft.fft(self.signal) / n
        self.fft_freq = np.fft.fftfreq(n, 1 / self.sampling_rate)
        
        # Only return positive frequencies
        pos_mask = self.fft_freq >= 0
        return self.fft_freq[pos_mask], self.fft_spectrum[pos_mask]
    
    def compute_wavelet_transform(self, wavelet='morl', scales=None):
        """
        Compute the Continuous Wavelet Transform of the signal.
        
        Parameters:
        -----------
        wavelet : str
            Wavelet to use (default: 'morl' for Morlet wavelet)
        scales : array_like
            Scales to use for the wavelet transform
        
        Returns:
        --------
        tuple
            (scales, coefficients) arrays
        """
        if self.signal is None:
            self.generate_signal()
        
        # Default scales
        if scales is None:
            scales = np.arange(1, 128)
        
        # Compute CWT
        self.wavelet_scales = scales
        self.wavelet_coeffs = signal.cwt(self.signal, signal.morlet, scales)
        
        return self.wavelet_scales, self.wavelet_coeffs
    
    def compute_laurent_series(self, z0=0, inner_order=5, outer_order=5):
        """
        Compute the Laurent series expansion of the signal around z0.
        
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
        if self.signal is None:
            self.generate_signal()
        
        # Convert signal to complex domain
        # This is a simplified approach - in a real application, you would use the analytic signal
        analytic_signal = signal.hilbert(self.signal)
        
        # Initialize coefficients
        self.laurent_coeffs = [0] * (inner_order + outer_order + 1)
        
        # Use numerical integration to compute coefficients
        for n in range(-inner_order, outer_order + 1):
            # Compute contour integral
            num_points = 100
            theta = np.linspace(0, 2*np.pi, num_points)
            r = 0.5  # Radius of contour
            
            integral = 0
            for t in theta:
                z = z0 + r * np.exp(1j*t)
                
                # Interpolate signal at z
                # This is a simplified approach - in a real application, you would use a proper interpolation
                idx = int(np.real(z) * self.sampling_rate) % len(self.signal)
                signal_val = analytic_signal[idx]
                
                integral += signal_val * np.exp(-1j*n*t) * 1j * r * np.exp(1j*t)
            
            integral *= 1 / (2*np.pi*1j) * (2*np.pi / num_points)
            self.laurent_coeffs[inner_order + n] = integral
        
        return self.laurent_coeffs
    
    def detect_faults_fft(self, threshold=0.2):
        """
        Detect faults using FFT analysis.
        
        Parameters:
        -----------
        threshold : float
            Threshold for fault detection
        
        Returns:
        --------
        list
            List of detected faults
        """
        if self.fft_freq is None or self.fft_spectrum is None:
            self.compute_fft()
        
        detected_faults = []
        
        # Find peaks in the spectrum
        peaks, _ = signal.find_peaks(np.abs(self.fft_spectrum), height=threshold*np.max(np.abs(self.fft_spectrum)))
        
        # Check if there are unexpected frequency components
        expected_freq = self.frequency
        for peak_idx in peaks:
            peak_freq = self.fft_freq[peak_idx]
            
            # Skip DC component and expected frequency
            if peak_freq < 0.1 or abs(peak_freq - expected_freq) < 0.1:
                continue
            
            # Check if it's a harmonic
            is_harmonic = False
            for i in range(2, 10):
                if abs(peak_freq - i * expected_freq) < 0.1:
                    is_harmonic = True
                    break
            
            if not is_harmonic:
                detected_faults.append(f"Unexpected frequency component at {peak_freq:.2f} Hz")
        
        # Check for amplitude changes
        main_peak_idx = np.argmin(np.abs(self.fft_freq - expected_freq))
        main_peak_amp = np.abs(self.fft_spectrum[main_peak_idx])
        
        if main_peak_amp < 0.5 * self.amplitude / 2:  # Expected amplitude in FFT is half the time domain amplitude
            detected_faults.append(f"Amplitude reduction at main frequency ({expected_freq:.2f} Hz)")
        
        return detected_faults
    
    def detect_faults_wavelet(self, threshold=0.7):
        """
        Detect faults using wavelet analysis.
        
        Parameters:
        -----------
        threshold : float
            Threshold for fault detection
        
        Returns:
        --------
        list
            List of detected faults
        """
        if self.wavelet_coeffs is None:
            self.compute_wavelet_transform()
        
        detected_faults = []
        
        # Compute scalogram (absolute value of wavelet coefficients)
        scalogram = np.abs(self.wavelet_coeffs)
        
        # Normalize
        scalogram = scalogram / np.max(scalogram)
        
        # Find sudden changes in the scalogram
        for i in range(1, scalogram.shape[1]):
            diff = np.abs(scalogram[:, i] - scalogram[:, i-1])
            if np.max(diff) > threshold:
                time_point = self.time[i]
                detected_faults.append(f"Sudden change detected at t = {time_point:.2f} s")
        
        return detected_faults
    
    def detect_faults_laurent(self, threshold=0.1):
        """
        Detect faults using Laurent series analysis.
        
        Parameters:
        -----------
        threshold : float
            Threshold for fault detection
        
        Returns:
        --------
        list
            List of detected faults
        """
        if self.laurent_coeffs is None:
            self.compute_laurent_series()
        
        detected_faults = []
        
        # Check for significant negative powers (poles)
        for i in range(len(self.laurent_coeffs) // 2):
            if abs(self.laurent_coeffs[i]) > threshold * np.max(np.abs(self.laurent_coeffs)):
                detected_faults.append(f"Potential pole detected (coefficient a_{-i})")
        
        return detected_faults
    
    def visualize_signal(self):
        """
        Visualize the signal in time domain.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        if self.signal is None:
            self.generate_signal()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.time, self.signal)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{self.signal_type.replace("_", " ").title()}, f = {self.frequency:.2f} Hz')
        ax.grid(True)
        
        # Mark fault if present
        if self.add_fault and self.fault_time is not None:
            ax.axvline(x=self.fault_time, color='r', linestyle='--', label=f'Fault ({self.fault_type})')
            ax.legend()
        
        return fig
    
    def visualize_fft(self):
        """
        Visualize the FFT spectrum.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        if self.fft_freq is None or self.fft_spectrum is None:
            self.compute_fft()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot magnitude spectrum
        ax.plot(self.fft_freq[self.fft_freq >= 0], np.abs(self.fft_spectrum[self.fft_freq >= 0]))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title('FFT Magnitude Spectrum')
        ax.grid(True)
        
        # Mark expected frequency
        ax.axvline(x=self.frequency, color='r', linestyle='--', label=f'Expected frequency ({self.frequency:.2f} Hz)')
        ax.legend()
        
        return fig
    
    def visualize_wavelet(self):
        """
        Visualize the wavelet transform.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        if self.wavelet_coeffs is None:
            self.compute_wavelet_transform()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot scalogram
        # Make sure the extent parameters match the actual dimensions
        extent = [0, self.duration, 1, len(self.wavelet_scales)]
        im = ax.imshow(np.abs(self.wavelet_coeffs), aspect='auto', 
                      extent=extent, cmap='viridis', origin='lower')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Scale')
        ax.set_title('Wavelet Scalogram')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Magnitude')
        
        # Mark fault if present
        if self.add_fault and self.fault_time is not None:
            ax.axvline(x=self.fault_time, color='r', linestyle='--', 
                      label=f'Fault ({self.fault_type})')
            ax.legend()
        
        return fig
    
    def visualize_laurent(self):
        """
        Visualize the Laurent series coefficients.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        if self.laurent_coeffs is None:
            self.compute_laurent_series()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Number of coefficients
        n_coeffs = len(self.laurent_coeffs)
        
        # Create index array centered at 0
        indices = np.arange(-n_coeffs//2, n_coeffs//2 + 1)
        
        # Plot coefficients
        ax.stem(indices, np.abs(self.laurent_coeffs))
        ax.set_xlabel('Power of z')
        ax.set_ylabel('Coefficient magnitude')
        ax.set_title('Laurent Series Coefficients')
        ax.grid(True)
        
        return fig