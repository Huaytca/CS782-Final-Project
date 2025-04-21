import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Transient detection function
def detect_transients(signal, fs, window_size=50, threshold=0.1):
    analytic_signal = hilbert(signal)
    phase = np.unwrap(np.angle(analytic_signal))
    phase_diff = np.diff(phase)
    avg_phase = np.convolve(phase_diff, np.ones(window_size)/window_size, 'same')
    diff2 = np.diff(avg_phase, n=2)
    diff2_pad = np.pad(diff2, (window_size//2, 0), mode='edge')
    transient_mask = diff2_pad > threshold
    edges = np.diff(transient_mask.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    
    # Handle edge cases
    if len(ends) < len(starts):
        ends = np.append(ends, len(signal) - 1)
    
    time = np.arange(len(signal)) / fs
    return [(time[s], time[e]) for s, e in zip(starts, ends)]

# Load sample signal
with open('Sample_Signal.txt', 'r') as file:
    data = np.loadtxt(file)

fs = 250e6  # Sampling frequency
time = np.arange(len(data)) / fs

# Detect transients
transients = detect_transients(data, fs, window_size=200, threshold=0.013)

# Plot original signal with transient markers
plt.figure(figsize=(12, 6))
plt.plot(time, data, label='Original Signal')
for start, end in transients:
    plt.axvline(start, color='lime', linestyle='--', linewidth=1, label='Transient Start')
    plt.axvline(end, color='magenta', linestyle='--', linewidth=1, label='Transient End')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal with Transient Start and End Points')
plt.legend()
plt.show()
