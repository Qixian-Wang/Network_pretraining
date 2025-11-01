import numpy as np

from analysismixin.truncate_data import TruncateDataMixIn
from visualization.propagation import plot_propagation_map

from config_file import configs

class PropagationAnalysisMixin(TruncateDataMixIn):
    def compute_propagation(self, config):
        data = self.firing_rate_result["spike_rates"]
        
        truncated_data = self.truncate_data(data, 0, configs.stim_interval)

        rate_threshold = config["rate_threshold"]

        for pattern_idx in range(configs.num_patterns):
            data_for_pattern = [truncated_data[i] for i in range(len(truncated_data)) if i % configs.num_patterns == pattern_idx]

            for data_idx, data_segment in enumerate(data_for_pattern):
                time_arrival_list = []
                for channel_data in data_segment:
                    mask = channel_data > rate_threshold
                    if np.any(mask):
                        time_index = np.argmax(mask)
                        time_delay = self.step_size * time_index
                    else:
                        time_delay = np.nan

                    time_arrival_list.append(time_delay)

                plot_propagation_map(self.figure_save_path, self.session_name, data_idx, pattern_idx, time_arrival_list)

