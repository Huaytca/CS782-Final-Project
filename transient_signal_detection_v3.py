'''
In this version, we don't have time as the x-axis, now its sample number
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def detect_transients(signal, window_size=50, threshold=0.1, margin_samples=10):
    """
    Phase-based transient detection with adjustable parameters and margin

    Parameters:
    -----------
    signal : array
        Input signal
    window_size : int
        Size of moving average window (larger = better for gradual transients)
    threshold : float
        Phase change threshold (smaller = more sensitive)
    margin_samples : int
        Extra samples to include before start and after end

    Returns:
    --------
    list of tuples
        (start_sample, end_sample) pairs for detected transients
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

    return [(start, end) for start, end in zip(starts, ends)]

# Load sample signal
with open('Sample_Signal.txt', 'r') as file:
    signal = np.loadtxt(file)

# Detect transients
transients = detect_transients(signal, window_size=150, threshold=0.02, margin_samples=40)

# Plot original signal with transient markers using sample numbers
plt.figure(figsize=(12, 6))
plt.plot(range(len(signal)), signal, label='Original Signal')

# Add shading for transient regions
for start, end in transients:
    plt.axvline(start, color='lime', linestyle='--', linewidth=1.5, label='Transient Start')
    plt.axvline(end, color='magenta', linestyle='--', linewidth=1.5, label='Transient End')
    plt.axvspan(start, end, color='cyan', alpha=0.2)

plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.title('Original Signal with Transient Detection (Sample Numbers)')

# Create legend without duplicate entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()

# Output detected transient ranges in sample numbers
print("Detected transient ranges (sample numbers):", transients)
