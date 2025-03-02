In this dataset, Bluetooth signals captured in a controlled setup are provided. There are several 
sampling frequencies (please, check sampling frequency for each data record). 
Each record file contains captured voltage signals as the output of high sampling rate oscilloscope.
 
Time-Series Data Generation
Each record can be converted to time series by using the following matlab script.
% This script reads an input record file from the proposed dataset, convert it into time series, 
% and plot the signal and also generates I/Q data.

%% Record file read
fileID = fopen ('Sample_Signal.txt','r'); % Reads the .txt file
data = fscanf(fileID,'%f');

%% Generating time series vector
fs = 250e6;                      % Sampling frequency
% fs must be entered 250e6, 5e9, 10e9 or 20e9 as per the record file sampling frequency.
N = length(data);              % Number of samples
Duration = N/fs;                 % Signal Duration
time = linspace(0,Duration,N);   % Time vector
Signal = [data time'];

%% Signal Plotting
figure,
plot(Signal(:,2),Signal(:,1));

%% I/Q Data Generation
% In-phase (I) and Quadrature(Q) components can be extracted by using discrete-time analytic signal 
% concept. In matlab, this could be quite easy. You may use the following script to extract I and Q 
% components.

HT = hilbert(Signal(:,1)); 
Q = imag(HT); % Quadrature (I) Data
I = real(HT); % In-phase (Q) Data
