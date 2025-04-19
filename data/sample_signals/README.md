# Sample Signals

This directory contains sample signals for testing and demonstration purposes in the AeroSim project.

## Available Signals

1. **clean_sine_wave.csv** - A pure sine wave with frequency 1 Hz
2. **noisy_sine_wave.csv** - A sine wave with added Gaussian noise
3. **sine_wave_impulse_fault.csv** - A sine wave with an impulse fault at t=2.5s
4. **sine_wave_step_fault.csv** - A sine wave with a step change at t=2.5s
5. **sine_wave_frequency_shift.csv** - A sine wave with a frequency shift at t=2.5s

## File Format

Each CSV file contains two columns:
- `time`: Time in seconds
- `amplitude`: Signal amplitude

## Generating New Signals

You can generate new sample signals by running the `generate_sample_signals.py` script:

```bash
python generate_sample_signals.py