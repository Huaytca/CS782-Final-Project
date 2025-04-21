import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Improved transient detection with adjustable parameters
def detect_transients(signal, fs, window_size=75, threshold=0.05, margin_samples=10):
    """
    Phase-based transient detection with adjustable parameters and margin
    
    Parameters:
    -----------
    signal : array
        Input signal
    fs : float
        Sampling frequency
    window_size : int
        Size of moving average window (larger = better for gradual transients)
    threshold : float
        Phase change threshold (smaller = more sensitive)
    margin_samples : int
        Extra samples to include before start and after end
    
    Returns:
    --------
    list of tuples
        (start_time, end_time) pairs for detected transients
    """
    analytic_signal = hilbert(signal)
    phase = np.unwrap(np.angle(analytic_signal))
    
    # Calculate phase differences and moving average
    phase_diff = np.diff(phase)
    avg_phase = np.convolve(phase_diff, np.ones(window_size)/window_size, 'same')
    
    # Find second derivative to detect changes in phase slope
    diff2 = np.diff(avg_phase, n=2)
    diff2_pad = np.pad(diff2, (window_size//2, 0), mode='edge')
    
    # Apply threshold for transient detection
    transient_mask = diff2_pad > threshold
    edges = np.diff(transient_mask.astype(int))
    
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    
    # Handle edge cases
    if len(ends) < len(starts):
        ends = np.append(ends, len(signal) - 1)
    
    # Apply margins to extend detected regions
    starts = np.maximum(starts - margin_samples, 0)
    ends = np.minimum(ends + margin_samples, len(signal) - 1)
    
    # Convert indices to time
    time = np.arange(len(signal)) / fs
    return [(time[s], time[e]) for s, e in zip(starts, ends)]

# Load sample signal
with open('Sample_Signal.txt', 'r') as file:
    data = np.loadtxt(file)

fs = 250e6  # Sampling frequency
time = np.arange(len(data)) / fs

# Detect transients with adjusted parameters
transients = detect_transients(
    data, 
    fs, 
    window_size=150,    # Increased from 50
    threshold=0.02,    # Decreased from 0.1
    margin_samples=40  # Added margin
)

# Plot original signal with transient markers
plt.figure(figsize=(12, 6))
plt.plot(time, data, label='Original Signal')

# Add shading for transient regions
for start, end in transients:
    plt.axvline(start, color='lime', linestyle='--', linewidth=1.5, label='Transient Start')
    plt.axvline(end, color='magenta', linestyle='--', linewidth=1.5, label='Transient End')
    plt.axvspan(start, end, color='cyan', alpha=0.2)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal with Full Transient Detection')

# Create legend without duplicate entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()
