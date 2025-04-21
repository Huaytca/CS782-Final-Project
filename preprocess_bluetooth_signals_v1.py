import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt
from module import load_data
import os

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
    elif dataset == 'B':
        fs = 10e9  # 10 Gsps
    elif dataset == 'C':
        fs = 20e9  # 20 Gsps
    else:  # Default to dataset D
        fs = 250e6  # 250 Msps
    
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
        if idx % 10 == 0:
            print(f"Processing signal {idx+1}/{len(df)}...")
            
        # Get the signal
        signal = row[signal_column]
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        # 1. Bandpass filter only for high-rate datasets (A/B/C)
        if dataset in ['A', 'B', 'C']:
            filtered_signal = bandpass_filter(signal, fs, dataset)
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

def bandpass_filter(signal, fs, dataset):
    """
    Bandpass filter implementation for removing spur signals
    as mentioned in Section 2.3 of the paper
    """
    nyquist = 0.5 * fs
    
    if dataset in ['A', 'B', 'C']:
        # High sampling rate datasets with direct sampling
        # "Remove spur signals at frequencies of 2.5 GHz or above"
        low = 2.4e9 / nyquist  # ISM2400 band lower bound
        high = 2.48e9 / nyquist  # Upper bound below spur signals
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

def normalize_signal(signal):
    """
    Normalize signal as mentioned in Section 2.3:
    "all the signals were normalized for scaling purposes"
    """
    if np.max(np.abs(signal)) > 0:
        return signal / np.max(np.abs(signal))
    return signal


def visualize_preprocessing(original_signal, filtered_signal, normalized_signal, fs, dataset='D'):
    """
    Visualize preprocessing steps in time and frequency domains
    """
    # Time vector
    N = len(original_signal)
    time = np.arange(N) / fs
    
    # Plot signals in time domain
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, original_signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 2)
    plt.plot(time, filtered_signal)
    plt.title('Filtered Signal (Removed Spur Signals)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 3)
    plt.plot(time, normalized_signal)
    plt.title('Normalized Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    # Plot frequency domain to check filter effectiveness
    plt.figure(figsize=(12, 8))
    
    # Compute FFT for original and filtered signals
    freq_original = np.fft.fftfreq(N, 1/fs)
    fft_original = np.abs(np.fft.fft(original_signal))
    fft_filtered = np.abs(np.fft.fft(filtered_signal))
    
    # Plot only positive frequencies
    positive_freq_mask = freq_original > 0
    freq_original = freq_original[positive_freq_mask]
    fft_original = fft_original[positive_freq_mask]
    fft_filtered = fft_filtered[positive_freq_mask]
    
    # Convert to MHz or GHz for easier reading
    freq_unit = 'MHz'
    freq_scaled = freq_original / 1e6  # MHz
    
    if dataset in ['A', 'B', 'C']:
        freq_unit = 'GHz'
        freq_scaled = freq_original / 1e9  # GHz
    
    plt.subplot(2, 1, 1)
    plt.plot(freq_scaled, 20*np.log10(fft_original + 1e-10))
    plt.title('Original Signal Spectrum')
    plt.xlabel(f'Frequency ({freq_unit})')
    plt.ylabel('Magnitude (dB)')
    
    plt.subplot(2, 1, 2)
    plt.plot(freq_scaled, 20*np.log10(fft_filtered + 1e-10))
    plt.title('Filtered Signal Spectrum')
    plt.xlabel(f'Frequency ({freq_unit})')
    plt.ylabel('Magnitude (dB)')
    
    plt.tight_layout()
    plt.show()

def visualize_iq_data(i_data, q_data):
    """
    Visualize I/Q data generated from Hilbert transform
    as described in Section 3 of the paper
    """
    plt.figure(figsize=(10, 10))
    
    # I/Q time domain plot
    plt.subplot(2, 1, 1)
    plt.plot(i_data, label='I (In-phase)')
    plt.plot(q_data, label='Q (Quadrature)')
    plt.title('I/Q Components')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # I/Q constellation plot
    plt.subplot(2, 1, 2)
    plt.scatter(i_data, q_data, alpha=0.5)
    plt.title('I/Q Constellation')
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    # Set the root directory
    root_directory = os.path.join(os.path.join(os.getcwd(), 'Bluetooth Datasets'), 'Dataset 5 Gsps')

    data = load_data(root_directory)

    preprocessed_data = preprocess_bluetooth_signals(data, signal_column='signal', dataset='A')
    print(preprocessed_data.columns)
    sampled_data = preprocessed_data.groupby('label').sample(n=20, random_state=42)
    fs = 5e9 # For Dataset A
    for idx, row in sampled_data.iterrows():
        original_signal = row['signal']
        filtered_signal = row['filtered_signal']
        normalized_signal = row['normalized_signal']
        visualize_preprocessing(original_signal, filtered_signal, normalized_signal, fs, dataset='A')
        i_data = row['I_data']
        q_data = row['Q_data']
        visualize_iq_data(i_data, q_data)
    # Visualize preprocessing steps
    # fs = 250e6  # For Dataset D
    # visualize_preprocessing(original_signal, filtered_signal, normalized_signal, fs, dataset='D')

    # Visualize I/Q data
    # i_data = preprocessed_data.iloc[signal_idx]['I_data']
    # q_data = preprocessed_data.iloc[signal_idx]['Q_data']
    # visualize_iq_data(i_data, q_data)
