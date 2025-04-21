import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt

# Function to load data
def load_data(root_dir):
    X = []
    y = []
    filenames = []
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
                filenames.append(file)
    
    df = pd.DataFrame({'signal': X, 'label': y, 'filename': filenames})
    return df

def my_improved_bayesian_change_point_detection(signal, window_size=15, overlap=0.5, start_threshold=2, end_threshold=0):
    """
    Implementation of Improved Bayesian Change Point Detection from the paper
    
    Parameters:
    -----------
    signal : array-like
        Analytic signal
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

    if isinstance(signal, pd.Series):
        signal = signal.values

    signal_magnitude = np.abs(signal)

    # Step 1: Divide signal into overlapping windows
    step = int(window_size * (1 - overlap))
    num_windows = (len(signal_magnitude) - window_size) // step + 1
    windows = [signal_magnitude[i * step:i * step + window_size] for i in range(num_windows)]
    
    # Step 2: Calculate window sums (fractal trajectory approximation)
    # Paper says: "compute summation of fractal trajectory entries"
    window_sums = [np.sum(window) for window in windows]
    
    # Step 3: Calculate difference vector between consecutive sums
    diff_vector = np.diff(window_sums)
    diff_vector -= 0.75 # TODO: See if you can remove this, but not right now cause its not causing problems i think hopefully maybe
    
    # Step 4: Find transient regions based on sign pattern
    # "The noise and steady parts have oscillations that make the differences 
    # contain positive and negative values in these portions, whereas the 
    # differences in the transient portion contain only positive values."
    start_idx, end_idx = None, None
    
    # Find first sequence of consecutive positive values (transient start)
    for i in range(len(diff_vector) - 2):
        if diff_vector[i] > start_threshold and diff_vector[i+1] > start_threshold and diff_vector[i+2] > start_threshold:
            start_idx = (i + 1) * step
            break
            
    # Find where positive sequence ends (transient end)
    if start_idx is not None:
        start_window = start_idx // step
        for i in range(start_window + 3, len(diff_vector) - 1):
            # if diff_vector[i] <= 0 or diff_vector[i+1] <= 0:
            if diff_vector[i] <= end_threshold:
                end_idx = i * step
                break
    # TODO: Figure out what signals don't have a transient and then remove them            
    if start_idx is None:
        start_idx = 0

    # Handle case where end wasn't found
    if end_idx is None and start_idx is not None:
        end_idx = len(signal) - 1
    
    return start_idx, end_idx, window_sums, diff_vector

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

def process_bluetooth_signal(signal, window_size, overlap, threshold):
    start_idx, end_idx, window_sums, diff_vector = my_improved_bayesian_change_point_detection(
        signal, window_size, overlap, threshold)
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

def bandpass_filter(signal, fs, dataset, low_cutoff, high_cutoff):
    """
    Bandpass filter implementation for removing spur signals
    as mentioned in Section 2.3 of the paper
    """
    nyquist = 0.5 * fs
    
    if dataset in ['A', 'B', 'C']:
        # High sampling rate datasets with direct sampling
        # "Remove spur signals at frequencies of 2.5 GHz or above"
        low = low_cutoff / nyquist  # ISM2400 band lower bound, was 2.35e9
        high = high_cutoff / nyquist  # Upper bound below spur signals
    else:  # Dataset D
        # RF front end with 250 Msps
        # "The bandwidth was tuned to 15 MHz-100 MHz"
        low = 15e6 / nyquist
        high = 100e6 / nyquist
    
    # Ensure normalized frequencies are in range [0, 1]
    low = min(max(low, 0.001), 0.99)
    high = min(max(high, 0.001), 0.99)
    
    # Apply bandpass filter (standard digital filter mentioned in paper)
    b, a = butter(5, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    
    return filtered

def preprocess_bluetooth_signals(df, signal_column='signal', dataset='D'):
    """
    Preprocess Bluetooth signals according to the paper's Section 2.3 methodology:
    1. Apply bandpass filter to remove spur signals (keeping only ISM2400 band)
    2. Normalize all signals for scaling purposes
    3. Apply Hilbert transform to get analytic signals and I/Q data
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing Bluetooth signal data
    signal_column : str
        Name of the column containing signal data as pandas Series
    dataset : str
        Dataset identifier ('A', 'B', 'C', or 'D')
        A: 5 Gsps, B: 10 Gsps, C: 20 Gsps, D: 250 Msps (default)
    """
    # Set sampling frequency based on dataset
    if dataset == 'A':
        fs = 5e9  # 5 Gsps
        low_cutoff = 2.00e9 # was 2.10e9
        high_cutoff = 2.80e9 # was 2.70e9
    elif dataset == 'B':
        fs = 10e9  # 10 Gsps
        low_cutoff = 2.35e9
        high_cutoff = 2.48e9
    elif dataset == 'C':
        fs = 20e9  # 20 Gsps
        low_cutoff = 2.35e9
        high_cutoff = 2.48e9
    else:  # Default to dataset D
        fs = 250e6  # 250 Msps
        low_cutoff = 2.35e9
        high_cutoff = 2.48e9
    
    # Create a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Add new columns for processed signals
    processed_df['filtered_signal'] = None
    processed_df['normalized_signal'] = None
    processed_df['analytic_signal'] = None
    processed_df['I_data'] = None  # In-phase component
    processed_df['Q_data'] = None  # Quadrature component
    
    print(f"Preprocessing {len(df)} signals from dataset {dataset}...")
    
    # Apply preprocessing to each signal
    for idx, row in processed_df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing signal {idx+1}/{len(df)}...")
            
        # Get the signal
        signal = row[signal_column]
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        # 1. Bandpass filter only for high-rate datasets (A/B/C)
        if dataset in ['B', 'C']:# was ['A', 'B', 'C']
            filtered_signal = bandpass_filter(signal, fs, dataset, low_cutoff, high_cutoff)
            filtered_signal = filtered_signal[:-int(filtered_signal.size * 0.05)] # Spoiler alert: it be helping
        else:
            filtered_signal = signal  # No filtering for Dataset D
        
        # 2. Normalize signal for scaling purposes
        normalized_signal = normalize_signal(filtered_signal)
        
        # 3. Apply Hilbert transform to get analytic signal (Section 3)
        analytic_signal = hilbert(normalized_signal)
        
        # 4. Extract I/Q components
        i_data = np.real(analytic_signal)
        q_data = np.imag(analytic_signal)
        
        # Store processed signals
        processed_df.at[idx, 'filtered_signal'] = filtered_signal
        processed_df.at[idx, 'normalized_signal'] = normalized_signal
        processed_df.at[idx, 'analytic_signal'] = analytic_signal
        processed_df.at[idx, 'I_data'] = i_data
        processed_df.at[idx, 'Q_data'] = q_data
    
    print("Preprocessing complete!")
    return processed_df

def normalize_signal(signal):
    """
    Normalize signal as mentioned in Section 2.3:
    "all the signals were normalized for scaling purposes"
    """
    if np.max(np.abs(signal)) > 0:
        return signal / np.max(np.abs(signal))
    return signal
