'''
In this version, we use Bayesian change point detection with my own algorithm
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from module import load_data
import os

def my_bayesian_change_point_detection(signal, window_size=15, overlap=0.5):
    """
    Implementation of Improved Bayesian Change Point Detection from the paper
    
    Parameters:
    -----------
    signal : array-like
        Input RF signal
    window_size : int
        Size of windows for fractal trajectory calculation
    overlap : float
        Overlap between consecutive windows (0-1)
        
    Returns:
    --------
    start_idx : int
        Index where transient starts
    end_idx : int
        Index where transient ends
    """
    # Step 1: Divide signal into overlapping windows
    step = int(window_size * (1 - overlap))
    num_windows = (len(signal) - window_size) // step + 1
    windows = [signal[i * step:i * step + window_size] for i in range(num_windows)]
    
    # Step 2: Calculate window sums (fractal trajectory approximation)
    # Paper says: "compute summation of fractal trajectory entries"
    window_sums = [np.sum(np.abs(window)) for window in windows]
    
    # Step 3: Calculate difference vector between consecutive sums
    diff_vector = np.diff(window_sums)

    diff_vector -= 0.005

    max_idx = np.argmax(diff_vector)
    
    # Step 4: Find transient regions based on sign pattern
    # "The noise and steady parts have oscillations that make the differences 
    # contain positive and negative values in these portions, whereas the 
    # differences in the transient portion contain only positive values."
    start_idx, end_idx = None, None
    
    # Find first sequence of consecutive positive values (transient start)
    for i in range(max_idx - 8, len(diff_vector) - 2):
        if diff_vector[i] > 0 and diff_vector[i+1] > 0 and diff_vector[i+2] > 0:
            start_idx = i * step
            break
            
    # Find where positive sequence ends (transient end)
    if start_idx is not None:
        start_window = start_idx // step
        for i in range(start_window + 3, len(diff_vector) - 2):
            # if diff_vector[i] <= 0 or diff_vector[i+1] <= 0:
            if diff_vector[i] <= 0:
                end_idx = i * step
                break
                
    # Handle case where end wasn't found
    if end_idx is None and start_idx is not None:
        end_idx = len(signal) - 1
    
    return start_idx, end_idx, window_sums, diff_vector

def improved_bayesian_change_point_detection(signal, window_size=15, overlap=0.5):
    """
    Implementation of Improved Bayesian Change Point Detection from the paper
    
    Parameters:
    -----------
    signal : array-like
        Input RF signal
    window_size : int
        Size of windows for fractal trajectory calculation
    overlap : float
        Overlap between consecutive windows (0-1)
        
    Returns:
    --------
    start_idx : int
        Index where transient starts
    end_idx : int
        Index where transient ends
    """
    # Step 1: Divide signal into overlapping windows
    step = int(window_size * (1 - overlap))
    num_windows = (len(signal) - window_size) // step + 1
    windows = [signal[i * step:i * step + window_size] for i in range(num_windows)]
    
    # Step 2: Calculate window sums (fractal trajectory approximation)
    # Paper says: "compute summation of fractal trajectory entries"
    window_sums = [np.sum(np.abs(window)) for window in windows]
    
    # Step 3: Calculate difference vector between consecutive sums
    diff_vector = np.diff(window_sums)
    
    # Step 4: Find transient regions based on sign pattern
    # "The noise and steady parts have oscillations that make the differences 
    # contain positive and negative values in these portions, whereas the 
    # differences in the transient portion contain only positive values."
    start_idx, end_idx = None, None
    
    # Find first sequence of consecutive positive values (transient start)
    for i in range(len(diff_vector) - 2):
        if diff_vector[i] > 0 and diff_vector[i+1] > 0 and diff_vector[i+2] > 0:
            start_idx = i * step
            break
            
    # Find where positive sequence ends (transient end)
    if start_idx is not None:
        start_window = start_idx // step
        for i in range(start_window + 3, len(diff_vector) - 2):
            # if diff_vector[i] <= 0 or diff_vector[i+1] <= 0:
            if diff_vector[i] <= 0:
                end_idx = i * step
                break
                
    # Handle case where end wasn't found
    if end_idx is None and start_idx is not None:
        end_idx = len(signal) - 1
    
    return start_idx, end_idx, window_sums, diff_vector

# Function to visualize results
def plot_transient_detection(signal, start_idx, end_idx):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(signal)), signal, label='Original Signal')
    
    # Mark start and end points
    plt.axvline(start_idx, color='lime', linestyle='--', linewidth=1.5, label='Transient Start')
    plt.axvline(end_idx, color='magenta', linestyle='--', linewidth=1.5, label='Transient End')
    
    # Shade transient region
    plt.axvspan(start_idx, end_idx, color='cyan', alpha=0.2)
    
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.title('Original Signal with Transient Detection')
    plt.legend()
    plt.show()
    
    print(f"Detected transient: Start={start_idx}, End={end_idx}")

# Example usage with a Bluetooth signal
def process_bluetooth_signal(signal):
    start_idx, end_idx, window_sums, diff_vector = improved_bayesian_change_point_detection(
        signal, window_size=200, overlap=0.65) # 150, 0.85; 150, 0.25
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(signal)
    # Plot the results
    plot_transient_detection(signal, start_idx, end_idx)
    
    # Also plot the fractal trajectory and difference vector
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(window_sums)
    plt.title('Window Sums (Fractal Trajectory)')
    plt.ylabel('Sum Value')
    
    plt.subplot(2, 1, 2)
    plt.plot(diff_vector)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.title('Difference Vector')
    plt.xlabel('Window Number')
    plt.ylabel('Difference')
    
    plt.tight_layout()
    plt.show()
    
    return start_idx, end_idx

if __name__ == '__main__':
    # Set the root directory
    root_directory = os.path.join(os.path.join(os.getcwd(), 'Bluetooth Datasets'), 'Dataset 250 Msps')
    nan_counter = 0
    data = load_data(root_directory)
    sampled_data = data.groupby('label').sample(n=20, random_state=42)
    for idx, row in sampled_data.iterrows():# changed to sampled_data.iterrows() if you want to look at them
        signal = row['signal']
        start_idx, end_idx, window_sums, diff_vector = improved_bayesian_change_point_detection(
        signal, window_size=200, overlap=0.70) # 200, 0.65
        process_bluetooth_signal(signal)
        if start_idx is None:
            print(row['label'])
            # process_bluetooth_signal(signal)
            nan_counter += 1
    print(nan_counter)
    # with open('Sample_Signal.txt', 'r') as file:
    #     signal = np.loadtxt(file)
    #     process_bluetooth_signal(signal)