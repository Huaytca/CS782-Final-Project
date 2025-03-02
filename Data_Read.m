% This script reads an example of data from the proposed dataset. 
% Also, plots the signal and generates I/Q data.

%% Text file read
fileID = fopen ('Sample_Signal.txt','r'); % Reads the .txt file
data = fscanf(fileID,'%f');

%% Generating time vector
fs = 250e6;                      % Sampling frequency
%With respect to loaded data from the dataset, fs must be one of the 250e6, 5e9, 10e9 or 20e9.
N = length(data);              % Number of samples
Duration = N/fs;                 % Signal Duration
time = linspace(0,Duration,N);   % Time vector
Signal = [data time'];

%% Signal Plotting
figure,
plot(Signal(:,2),Signal(:,1));

%% In-phase and Quadrature Data Generation
HT = hilbert(Signal(:,1)); 
Q = imag(HT); % Quadrature Data
I = real(HT); % In-phase Data
