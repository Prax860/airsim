import numpy as np
import matplotlib.pyplot as plt
import os

def generate_sine_wave(frequency=1.0, amplitude=1.0, duration=5.0, sampling_rate=100, 
                      noise_level=0.1, add_fault=False, fault_type=None, fault_time=None):
    """Generate a sine wave with optional noise and fault."""
    time = np.linspace(0, duration, int(duration * sampling_rate))
    signal = amplitude * np.sin(2 * np.pi * frequency * time)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * amplitude, len(time))
        signal += noise
    
    # Add fault
    if add_fault and fault_type is not None and fault_time is not None:
        fault_idx = np.argmin(np.abs(time - fault_time))
        
        if fault_type == "impulse":
            signal[fault_idx] += 5 * amplitude
        elif fault_type == "step_change":
            signal[fault_idx:] += 0.5 * amplitude
        elif fault_type == "frequency_shift":
            new_freq = 2 * frequency
            signal[fault_idx:] = amplitude * np.sin(2 * np.pi * new_freq * (time[fault_idx:] - time[fault_idx]) + 
                                                  2 * np.pi * frequency * time[fault_idx])
    
    return time, signal

def save_signal(time, signal, filename, title=None):
    """Save signal to a file and create a plot."""
    # Save data
    data = np.column_stack((time, signal))
    np.savetxt(filename + '.csv', data, delimiter=',', header='time,amplitude', comments='')
    
    # Create and save plot
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    if title:
        plt.title(title)
    plt.grid(True)
    plt.savefig(filename + '.png', dpi=150)
    plt.close()

def main():
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate clean sine wave
    time, signal = generate_sine_wave(frequency=1.0, amplitude=1.0, duration=5.0, 
                                     sampling_rate=100, noise_level=0.0)
    save_signal(time, signal, os.path.join(output_dir, 'clean_sine_wave'), 
               'Clean Sine Wave (1 Hz)')
    
    # Generate noisy sine wave
    time, signal = generate_sine_wave(frequency=1.0, amplitude=1.0, duration=5.0, 
                                     sampling_rate=100, noise_level=0.2)
    save_signal(time, signal, os.path.join(output_dir, 'noisy_sine_wave'), 
               'Noisy Sine Wave (1 Hz)')
    
    # Generate sine wave with impulse fault
    time, signal = generate_sine_wave(frequency=1.0, amplitude=1.0, duration=5.0, 
                                     sampling_rate=100, noise_level=0.1, 
                                     add_fault=True, fault_type="impulse", fault_time=2.5)
    save_signal(time, signal, os.path.join(output_dir, 'sine_wave_impulse_fault'), 
               'Sine Wave with Impulse Fault at t=2.5s')
    
    # Generate sine wave with step change fault
    time, signal = generate_sine_wave(frequency=1.0, amplitude=1.0, duration=5.0, 
                                     sampling_rate=100, noise_level=0.1, 
                                     add_fault=True, fault_type="step_change", fault_time=2.5)
    save_signal(time, signal, os.path.join(output_dir, 'sine_wave_step_fault'), 
               'Sine Wave with Step Change at t=2.5s')
    
    # Generate sine wave with frequency shift fault
    time, signal = generate_sine_wave(frequency=1.0, amplitude=1.0, duration=5.0, 
                                     sampling_rate=100, noise_level=0.1, 
                                     add_fault=True, fault_type="frequency_shift", fault_time=2.5)
    save_signal(time, signal, os.path.join(output_dir, 'sine_wave_frequency_shift'), 
               'Sine Wave with Frequency Shift at t=2.5s')
    
    print("Sample signals generated successfully.")

if __name__ == "__main__":
    main()