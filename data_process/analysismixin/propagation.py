import numpy as np

from analysismixin.truncate_data import TruncateDataMixIn
from visualization.propagation import plot_propagation_map

class PropagationAnalysisMixin(TruncateDataMixIn):
    def compute_propagation(self, config):
        analysis_duration = config["max_time_delay"]
        rate_threshold = config["rate_threshold"]

        data = self.firing_rate_result["spike_rates"]
        truncated_data = self.truncate_data(data, 0, analysis_duration)

        for pattern_idx in range(self.num_patterns):
            data_for_pattern = [truncated_data[i] for i in range(len(truncated_data)) if i % self.num_patterns == pattern_idx]

            for data_idx, data_segment in enumerate(data_for_pattern):
                time_arrival_list = []
                max_firing_rate_list = []
                for channel_data in data_segment:
                    mask = channel_data > rate_threshold
                    if np.any(mask):
                        time_index = np.argmax(mask)
                        time_delay = self.step_size * time_index
                    else:
                        time_delay = np.nan

                    time_arrival_list.append(time_delay if time_delay < analysis_duration else np.nan)
                    max_firing_rate_list.append(np.max(channel_data))

                plot_propagation_map(self.figure_save_path, self.session_name, data_idx, pattern_idx, time_arrival_list, max_firing_rate_list)

