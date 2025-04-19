import unittest
import numpy as np
from modules.signal_analysis import SignalAnalyzer

class TestSignalAnalyzer(unittest.TestCase):
    
    def setUp(self):
        # Create an analyzer instance for testing
        self.analyzer = SignalAnalyzer(signal_type="sine_wave", frequency=1.0, 
                                      amplitude=1.0, duration=5.0, sampling_rate=100)
    
    def test_initialization(self):
        # Test that the analyzer initializes correctly
        self.assertEqual(self.analyzer.signal_type, "sine_wave")
        self.assertEqual(self.analyzer.frequency, 1.0)
        self.assertEqual(self.analyzer.amplitude, 1.0)
        self.assertEqual(self.analyzer.duration, 5.0)
        self.assertEqual(self.analyzer.sampling_rate, 100)
    
    def test_generate_signal(self):
        # Test signal generation
        time, signal = self.analyzer.generate_signal()
        
        # Check dimensions
        self.assertEqual(len(time), 500)  # duration * sampling_rate
        self.assertEqual(len(signal), 500)
        
        # Check time range
        self.assertAlmostEqual(time[0], 0.0)
        self.assertAlmostEqual(time[-1], 5.0, places=2)
        
        # For sine wave, check amplitude
        peak = np.max(signal)
        trough = np.min(signal)
        self.assertAlmostEqual(peak, 1.0, places=1)
        self.assertAlmostEqual(trough, -1.0, places=1)
    
    def test_compute_fft(self):
        # Test FFT computation
        freq, spectrum = self.analyzer.compute_fft()
        
        # Check dimensions
        self.assertEqual(len(freq), 251)  # (duration * sampling_rate) // 2 + 1
        self.assertEqual(len(spectrum), 251)
        
        # Check frequency range
        self.assertAlmostEqual(freq[0], 0.0)
        self.assertAlmostEqual(freq[-1], 50.0, places=0)  # Nyquist frequency
        
        # For sine wave, check peak at expected frequency
        peak_idx = np.argmax(np.abs(spectrum[1:]))  # Skip DC component
        self.assertAlmostEqual(freq[peak_idx+1], 1.0, places=1)
    
    def test_fault_detection(self):
        # Test with a fault
        analyzer_with_fault = SignalAnalyzer(signal_type="sine_wave", frequency=1.0, 
                                           amplitude=1.0, duration=5.0, sampling_rate=100,
                                           add_fault=True, fault_type="frequency_shift", 
                                           fault_time=2.5)
        
        # Generate signal with fault
        time, signal = analyzer_with_fault.generate_signal()
        
        # Detect faults
        faults = analyzer_with_fault.detect_faults_fft()
        
        # There should be at least one fault detected
        self.assertTrue(len(faults) > 0)

if __name__ == '__main__':
    unittest.main()