import numpy as np
import numba as nb
from miv.core.datatype.spikestamps import Spikestamps

from visualization.firing_rate import plot_phase_firing_rate


@nb.njit
def _spike_count(spike_times, start_indices, end_indices, window_centers, gaussian_sigma, norm):
    weights = np.zeros(len(start_indices))
    for i in range(len(start_indices)):
        distances = spike_times[start_indices[i]:end_indices[i]] - window_centers[i]
        weight = norm * np.exp(-0.5 * (distances / gaussian_sigma) ** 2)
        weights[i] = np.sum(weight)
    return weights

@nb.njit
def _remove_artifacts(spike_times, start_indices, end_indices):
    artifact_mask = np.ones(len(spike_times))
    for i in range(len(start_indices)):
        artifact_mask[start_indices[i]:end_indices[i]] = 0
    return spike_times[artifact_mask == 1]


class MovingAverageFiringRateMixIn:

    def compute_firing_rate(self, config):
        pre_offset = config["pre_offset"]
        post_offset = config["post_offset"]
        window_size = config["window_size"]
        step_size = config["step_size"]
        gaussian_sigma = config["gaussian_sigma"]
        method = config["method"]
        plot_firing_rate = config["plot_firing_rate"]

        self.step_size = step_size
        self.pre_offset = pre_offset
        self.post_offset = post_offset
        self.num_channels = len(self.spike_train)
        self.spike_train = Spikestamps(self._remove_artifacts())

        print(f"Number of channels to read: {len(self.spike_train)}", flush=True)

        if method == "moving_average":
            self.firing_rate_result = self._moving_average_firing_rate(
                window_size, gaussian_sigma
            )

        elif method == "instantaneous":
            self.firing_rate_result = self._instantaneous_firing_rate(
                step_size
            )

        if plot_firing_rate:
            plot_phase_firing_rate(self.figure_save_path, self.session_name, self.spike_train, self.firing_rate_result, self.step_size)            

        return self
    
    def _remove_artifacts(self):
        if self.stimulation_init_time is None:
            return self.spike_train

        stim_times = np.asarray(self.stimulation_init_time)
        spike_data_cleaned = []

        for channel_spike_times in self.spike_train:
            start_indices = np.searchsorted(channel_spike_times, stim_times - self.pre_offset, side='left')
            end_indices = np.searchsorted(channel_spike_times, stim_times + self.post_offset, side='right')

            spike_data_cleaned.append(_remove_artifacts(channel_spike_times, start_indices, end_indices))

        return spike_data_cleaned
    

    def _moving_average_firing_rate(self, window_size, gaussian_sigma):
        window_starts = np.arange(0, self.spike_train.get_last_spikestamp() - window_size + self.step_size, self.step_size)
        window_centers = window_starts + window_size / 2
        num_windows = len(window_starts)
                
        spike_rates_matrix = np.zeros((self.num_channels, num_windows))

        norm = 1.0 / (np.sqrt(2.0 * np.pi) * gaussian_sigma)

        for ch_idx, spike_times in enumerate(self.spike_train):
            start_indices = np.searchsorted(spike_times, window_starts, side='left')
            end_indices = np.searchsorted(spike_times, window_starts + window_size, side='right')
            weights = _spike_count(spike_times, start_indices, end_indices, window_centers, gaussian_sigma, norm)


            spike_rates_matrix[ch_idx, :] = weights

        return {
            'spike_rates': spike_rates_matrix,
            'window_centers': window_centers
        }
    
    def _instantaneous_firing_rate(self, step_size):
        window_starts = np.arange(0, self.spike_train.get_last_spikestamp(), self.step_size)
        window_centers = window_starts + step_size / 2
        num_windows = len(window_starts)
                
        spike_rates_matrix = np.zeros((self.num_channels, num_windows))
        
        for ch_idx, spike_times in enumerate(self.spike_train):
            start_indices = np.searchsorted(spike_times, window_starts, side='left')
            end_indices = np.searchsorted(spike_times, window_starts + step_size, side='right')
            spike_counts = end_indices - start_indices
            spike_rates_matrix[ch_idx, :] = spike_counts / step_size
        
        return {
            'spike_rates': spike_rates_matrix,
            'window_centers': window_centers
        }
