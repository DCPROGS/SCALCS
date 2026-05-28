"""
scsim.py - Module for simulating single ion channel recordings

This module provides functions to:
1. Simulate state transitions and intervals
2. Sample the data at regular intervals to create realistic recordings
3. Add noise and filtering to mimic experimental conditions
4. Analyze and visualize the results

Author: Remis Lape
Date: February 27, 2025
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Union
from scipy import signal

from samples import samples
from scalcs import qmatlib as qml

class SCSimulator(qml.QMatrix):
    """Simulates single ion channel recordings."""
    def __init__(self, mec):
        super().__init__(mec)

    def sanity_check(self, state: int) -> None:
        """ Validate inputs before simulation starts. """
        if not hasattr(self, 'Q') or not hasattr(self, 'kA'):
            raise AttributeError("Model object must have Q matrix and kA attributes")
        if not isinstance(self.Q, np.ndarray) or self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q matrix must be a square 2D numpy array")
        if not 0 <= state < self.Q.shape[0]:
            raise ValueError(f"Initial state {state} must be between 0 and {self.Q.shape[0]-1}")

    def simulate_intervals(self, tres: float, state: int, opamp: float = 5.0, 
                        nintmax: int = 5000, seed: Optional[int] = None) -> None:
        """
        Simulates single channel state intervals based on a Markov model.
        
        Parameters:
        -----------
        tres : float
            Time resolution in seconds. Intervals shorter than this are merged
        state : int
            Initial state of the channel
        opamp : float, optional
            Open channel amplitude (default=5)
        nintmax : int, optional
            Maximum number of intervals to simulate (default=5000)
        seed : Optional[int], optional
            Random seed for reproducibility (default=None)
        
        Generates:
        --------
            - Array of interval durations (seconds)
            - Array of amplitudes for each interval
            - Array of flags (always zeros in current implementation)
            - Number of state transitions that occurred
        """

        self.opamp = opamp
        self.starting_state = state
        if seed is not None:
            random.seed(seed) # Set random seed if provided
        self.sanity_check(state)
        
        picum = np.cumsum(self.transition_probability(), axis=1) # Cumulative transition probability
        tmean = self.state_lifetimes() # Mean lifetime in each state
                
        # Initialize counters and state
        nint, ntrns = 0, 0
        current_state = state
        a = opamp if current_state < self.kA else 0.0 # Determine amplitude of initial state (open or closed)
        # Generate initial interval duration using exponential distribution
        t = -tmean[current_state] * math.log(random.random()) if tmean[current_state] < float('inf') else float('inf')
        tints, ampls = [t], [a] # Lists to store intervals and amplitudes
        
        while nint < nintmax - 1: # Main simulation loop
            newst, t, a = self.next_state(current_state, picum, tmean) # Get next state, its lifetime and amplitude
            ntrns += 1
            
            # Handle intervals shorter than time resolution
            if t < tres:
                tints[-1] += t # Merge with previous interval (add time but keep amplitude)
            else:
                # Check if amplitude changed
                if ((a != 0 and ampls[-1] != 0) or (a == 0 and ampls[-1] == 0)):
                    tints[-1] += t # Same state type (open or closed), extend previous interval
                else:
                    # State type changed, finish current interval and new interval will start
                    tints.append(t)
                    ampls.append(a)
                    nint += 1
            
            current_state = newst # Update current state
    
        # Convert lists to numpy arrays for efficiency
        self.tints = np.array(tints)
        self.ampls = np.array(ampls)
        self.flags = np.zeros(len(tints), dtype='bool')
        self.ntrns = ntrns

    def next_state(self, present: int, picum: np.ndarray, tmean: np.ndarray) -> Tuple[int, float, float]:
        """
        Determines the next state, its lifetime and amplitude.
        
        Parameters:
        -----------
        present : int
            Current state index
        picum : np.ndarray
            Cumulative transition probability matrix
        tmean : np.ndarray
            Array of mean lifetimes for each state
            
        Returns:
        --------
        Tuple[int, float, float]
            - Next state index
            - Duration of the next state (seconds)
            - Amplitude of the next state
        """
        # Find possible next states based on random draw
        r = random.random()
        possible = np.nonzero(picum[present] >= r)[0]
        # Remove the current state from possibilities
        possible_transitions = np.delete(possible, np.where(possible == present))
        
        if len(possible_transitions) == 0:
            # Handle rare case where no valid transition is found
            # Choose randomly among all states except the present one
            all_states = np.arange(len(tmean))
            possible_transitions = np.delete(all_states, present)
        
        # Select the next state (first in the filtered array)
        next_state = possible_transitions[0]
        # Generate lifetime using exponential distribution
        t = random.expovariate(1 / tmean[next_state])
        # Determine amplitude based on state type (open or closed)
        a = self.opamp if next_state < self.kA else 0
        
        return next_state, t, a

    def analyse_intervals(self) -> Dict[str, Union[float, int]]:
        """
        Analyzes the simulated intervals to extract basic statistics.
        
        Parameters:
        -----------
        tints : np.ndarray
            Array of interval durations
        ampls : np.ndarray
            Array of amplitudes
        opamp : float, optional
            Open channel amplitude
            
        Returns:
        --------
        Dict[str, Union[float, int]]
            Information about the simulation
        """
        # Define what counts as an open interval (within 20% of opamp)
        threshold = self.opamp * 0.8
        is_open = self.ampls >= threshold
        # Separate open and closed intervals
        self.open_intervals = self.tints[is_open]
        self.closed_intervals = self.tints[~is_open]
        # Calculate basic statistics
        total_duration = np.sum(self.tints)
        total_open_time = np.sum(self.open_intervals)
        
        return {
            "Starting state": self.mec.States[self.starting_state].name,
            "Total duration (s)": total_duration,
            "Number of intervals": len(self.tints),
            "Total transitions": self.ntrns,
            "Open intervals": len(self.open_intervals),
            "Closed intervals": len(self.closed_intervals),
            "Mean open time (ms)": np.mean(self.open_intervals) * 1000 if len(self.open_intervals) > 0 else 0,
            "Mean closed time (ms)": np.mean(self.closed_intervals) * 1000 if len(self.closed_intervals) > 0 else 0,
            "Open probability": total_open_time / total_duration if total_duration > 0 else 0,
            "Open frequency (Hz)": len(self.open_intervals) / total_duration if total_duration > 0 else 0
        }
        
    def sample_channel_trace(self, sampling_interval: float = 20e-6, duration: Optional[float] = None, 
                             noise_std: float = 0.2) -> Dict[str, Union[float, int]]:
        """
        Samples the single-channel current at regular intervals to create a realistic trace.
        
        Parameters:
        -----------
        sampling_interval : float, optional
            Time between samples in seconds (default: 20 microseconds)
        duration : float, optional
            Total duration to sample (in seconds). If None, uses the entire simulation.
        noise_std : float, optional
            Standard deviation of Gaussian noise to add (in pA)
            
        Returns:
        --------
        Dict[str, Union[float, int]]
            Information about the sampling
        """
        # Calculate total duration of the simulation
        total_duration = np.sum(self.tints)
        # If duration is specified, use the smaller of total_duration or specified duration
        if duration is not None:
            max_duration = min(total_duration, duration)
        else:
            max_duration = total_duration
        
        # Create time points at regular intervals
        self.time_points = np.arange(0, max_duration, sampling_interval)
        num_samples = len(self.time_points)
        self.current = np.zeros(num_samples) # Initialize current trace array
        interval_endpoints = np.cumsum(self.tints) # Convert intervals to cumulative time points
        last_endpoint = 0 # Start from time=0
        
        # Fill the current trace based on amplitudes
        for i, (endpoint, amplitude) in enumerate(zip(interval_endpoints, self.ampls)):
            # Find indices in the sample array that fall within this interval
            start_idx = np.searchsorted(self.time_points, last_endpoint, side='left')
            end_idx = np.searchsorted(self.time_points, endpoint, side='left')
            
            self.current[start_idx:end_idx] = amplitude # Set the current value for these indices
            last_endpoint = endpoint # Update last endpoint
            if endpoint >= max_duration: # Break if we've gone beyond our desired duration
                break
        
        if noise_std > 0: # Add Gaussian noise to mimic recording noise
            noise = np.random.normal(0, noise_std, num_samples)
            self.current += noise

        return {"Sampling frequency (Hz)": 1.0 / sampling_interval,
            "Total samples": len(self.time_points)}

    def apply_filter( self, cutoff_freq: float = 5000, order: int = 4) -> None: 
        """
        Applies a low-pass filter to the current trace to simulate bandwidth limitations
        of recording equipment.
        
        Parameters:
        -----------
        cutoff_freq : float, optional
            Cutoff frequency in Hz (default: 5000 Hz)
        order : int, optional
            Order of the Butterworth filter (default: 4)
            
        Generates:
        --------
        np.ndarray
            Filtered current trace
        """
        # Calculate sampling frequency
        sampling_freq = 1.0 / (self.time_points[1] - self.time_points[0])
        # Normalize cutoff frequency
        nyquist = 0.5 * sampling_freq
        normal_cutoff = cutoff_freq / nyquist
        # Design Butterworth low-pass filter
        b, a = signal.butter(order, normal_cutoff, btype='low')
        # Apply filter
        self.filtered_current = signal.filtfilt(b, a, self.current)

    def plot_channel_simulation(self, duration: Optional[float] = None) -> None:
        """
        Plots the simulated single channel time series as idealized steps.
        
        Parameters:
        -----------
        duration : Optional[float], optional
            Maximum duration to plot
        """

        # Convert intervals to time series
        t_points, a_points = [], []
        t_current = 0
        for i, (interval, amp) in enumerate(zip(self.tints, self.ampls)):
            t_points.extend([t_current, t_current + interval])
            a_points.extend([amp, amp])
            t_current += interval
            if duration and (t_current > duration): # Limit to specified duration
                break
        
        # Plot the channel activity
        plt.figure(figsize=(12, 6))
        plt.step(t_points, a_points, where='post')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (pA)')
        plt.title('Single Channel Simulation - Idealized Trace')
        plt.ylim(-1, max(self.ampls) + 1)
        plt.grid(True, alpha=0.3)
        if duration: # Limit x-axis if duration specified
            plt.xlim(0, duration)
        plt.tight_layout()

    def plot_sampled_trace(self, window: Optional[Tuple[float, float]] = None) -> None:
        """
        Plots the sampled current trace.
        
        Parameters:
        -----------
        window : Optional[Tuple[float, float]], optional
            Time window to display (start_time, end_time) in seconds
        """

        plt.figure(figsize=(12, 6))
        time_ms = self.time_points * 1000 # Convert time to ms for better readability
        plt.plot(time_ms, self.current, 'gray', alpha=0.5, label='Raw') # Plot raw trace
        if self.filtered_current is not None: # Plot filtered trace if provided
            plt.plot(time_ms, self.filtered_current, 'b', label='Filtered')
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (pA)')
        plt.title('Sampled Single-Channel Current Trace')
        if window: # Set view window if specified
            plt.xlim(window[0]*1000, window[1]*1000)  # Convert to ms
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def plot_amplitude_histogram(self, bins: int = 50) -> None:
        """
        Creates an all-points histogram of current amplitudes.
        
        Parameters:
        -----------
        bins : int, optional
            Number of histogram bins
        """

        plt.figure(figsize=(10, 6))
        plt.hist(self.current, bins=bins, alpha=0.7, color='gray', label='Raw') # Plot raw data histogram
        if self.filtered_current is not None: # Plot filtered data histogram if provided
            plt.hist(self.filtered_current, bins=bins, alpha=0.5, color='blue', label='Filtered')
        plt.xlabel('Current (pA)')
        plt.ylabel('Frequency')
        plt.title('All-Points Amplitude Histogram')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    def plot_all(self, duration=1) -> None:
        """ Plot all available plots. """
        self.plot_channel_simulation(duration) # Plot idealized trace
        self.plot_sampled_trace() # Plot sampled trace
        self.plot_amplitude_histogram() # Create histogram       
        plt.show()
        
# Example usage
if __name__ == "__main__":

    c = 1e-6 # 10 uM
    mec = samples.CH82()
    mec.set_eff('c', c)
    simulator = SCSimulator(mec)

    # Step 1: Simulate intervals
    simulator.simulate_intervals(
        tres=50e-6, 
        state=0, 
        opamp=5.0, 
        nintmax=10000, 
        seed=42)
    
    # Step 2: Analyze intervals
    simulation_result = simulator.analyse_intervals()
    # Print analysis results
    print("Simulation Results:")
    for key, value in simulation_result.items():
        print(f"{key}: {value}")

    # Step 3: Sample at regular intervals
    sampling_info = simulator.sample_channel_trace(
        sampling_interval=20e-6,  # 20 µs sampling (50 kHz)
        duration=1,        # 100 ms recording
        noise_std=1.0 #0.15,      # 0.15 pA noise 
    )
    print("Sampling Results:")
    for key, value in sampling_info.items():
        print(f"{key}: {value}")
    
    # Step 4: Apply filtering
    simulator.apply_filter(
        cutoff_freq=5000,    # 5 kHz filter
        order=4
    )

    # Step 5: Plot simulations
    simulator.plot_all()
    
    
