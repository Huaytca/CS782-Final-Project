import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Read the data
with open('Sample_Signal.txt', 'r') as file:
    data = np.loadtxt(file)

# Generating time vector
fs = 250e6  # Sampling frequency
N = len(data)  # Number of samples
Duration = N / fs  # Signal Duration
time = np.linspace(0, Duration, N)  # Time vector
Signal = np.column_stack((data, time))

# Signal Plotting
plt.figure()
plt.plot(Signal[:, 1], Signal[:, 0])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal Plot')
plt.show()

# In-phase and Quadrature Data Generation
HT = hilbert(Signal[:, 0])
Q = np.imag(HT)  # Quadrature Data
I = np.real(HT)  # In-phase Data
