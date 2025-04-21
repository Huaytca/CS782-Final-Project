import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def detect_transients(signal, window_size=75, threshold=0.05, margin_samples=20):
    """
    Phase-based transient detection optimized for Bluetooth RF signals
    """
    # Convert to numpy array if it's a pandas Series
    if isinstance(signal, pd.Series):
        signal = signal.values
        
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
    
    return [(int(start), int(end)) for start, end in zip(starts, ends)]

# Process each signal in the DataFrame
def process_bluetooth_signals(df, signal_column='signal'):
    """
    Process all Bluetooth signals in the DataFrame and visualize transients
    """
    all_transients = []
    
    for idx, row in df.iterrows():
        signal = row[signal_column]
        
        # Detect transients
        transients = detect_transients(signal, window_size=150, threshold=0.025, margin_samples=40)
        all_transients.append(transients)
        
        # Plot signal with detected transients
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(signal)), signal, label='Original Signal')
        
        # Add transient markers and shading
        for start, end in transients:
            plt.axvline(start, color='lime', linestyle='--', linewidth=1.5, label='Transient Start')
            plt.axvline(end, color='magenta', linestyle='--', linewidth=1.5, label='Transient End')
            plt.axvspan(start, end, color='cyan', alpha=0.2)
        
        plt.xlabel('Sample Number')
        plt.ylabel('Amplitude')
        plt.title(f'Bluetooth Signal {idx+1} with Transient Detection')
        
        # Create legend without duplicate entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.show()
        
        # Print detected transient ranges
        print(f"Signal {idx+1} - Detected transient ranges: {transients}")
    
    return all_transients

# Function to load data
def load_data(root_dir):
    X = []
    y = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    values = np.array([float(line.strip()) for line in f])
                X.append(pd.Series(values))           
                # Use folder name as label
                label = os.path.relpath(root, root_dir)
                y.append(label)
    
    df = pd.DataFrame({'signal': X, 'label': y})
    return df

if __name__ == '__main__':
    # Set the root directory
    root_directory = os.path.join(os.path.join(os.getcwd(), 'Bluetooth Datasets'), 'Dataset 250 Msps')

    data = load_data(root_directory)

    # Example usage:
    transients = process_bluetooth_signals(data)
