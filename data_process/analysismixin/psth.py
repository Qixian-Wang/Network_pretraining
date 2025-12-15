import numpy as np

from visualization.psth import plot_spike_train_with_psth
from config_file import configs

class PsthAnalysisMixin:
    def compute_psth(self, config):
        visualize_time_bias = config["visualize_time_bias"]
        visualize_duration = config["visualize_duration"]
        bin_size = config["bin_size"]
        similarity_duration = config["similarity_duration"]
        plot_psth_flag = config["plot_psth"]
        num_subplots = config["num_subplots"]

        if self.session_name.startswith("pretrain"):
            repetition = configs.pretrain_stimulation_repetition
        else:
            repetition = 1

        if num_subplots is not None:
            num_rows = num_subplots
        else:
            num_rows = len(self.stimulation_init_time) // self.num_patterns

        start_time_list = []
        end_time_list = []
        centers_list = []
        rates_list = []
        similarity_list = []

        for pattern_idx in range(self.num_patterns):
            aligned_data = []

            for row in range(num_rows):
                start_time = self.stimulation_init_time[row * self.num_patterns + pattern_idx] - visualize_time_bias
                end_time = start_time + visualize_duration

                start_time_list.append(start_time)
                end_time_list.append(end_time)

                aligned_data.append(self._collect_window_spikes(start_time, end_time))
            
            centers, trail_average_rates, channel_average_rates_list = self._compute_psth(aligned_data, window_len=visualize_duration, n_channels=len(self.spike_train), bin_size=bin_size)

            centers_list.append(centers)
            rates_list.append(trail_average_rates)

            similarity = self._compute_ingroup_similarity(channel_average_rates_list, visualize_time_bias, bin_size, similarity_duration=similarity_duration)
            similarity_list.append(similarity)
            
        self.global_similarity_list.append([self.session_name, similarity_list])

        if plot_psth_flag:
            plot_spike_train_with_psth(
                spike_train=self.spike_train,
                figure_save_path=self.figure_save_path,
                session_name=self.session_name,
                num_patterns=self.num_patterns,
                start_time_list=start_time_list,
                end_time_list=end_time_list,
                centers=centers_list,
                rates=rates_list,
                similarity_list=similarity_list,
                visualize_time_bias=visualize_time_bias,
                visualize_duration=visualize_duration,
                num_rows=num_rows,
                bin_size=bin_size,
                pattern_dict=self.pattern_dict,
                repetition=repetition,
                )
        
        return self


    def _collect_window_spikes(self, t0, t1):
        all_spikes = []
        for channel_data in self.spike_train.data:
            all_spikes.extend(channel_data)
        flat_spikes = np.array(sorted(all_spikes))
        spikes_in_window = (flat_spikes >= t0) & (flat_spikes <  t1)

        return flat_spikes[spikes_in_window] - t0


    def _compute_psth(self, aligned_data, window_len, n_channels, bin_size=0.01):
        edges = np.arange(0, window_len + 1e-12, bin_size)
        centers = (edges[:-1] + edges[1:]) / 2.0
        channel_average_rates_list = []

        for data in aligned_data:
            counts, _ = np.histogram(data, bins=edges)
            channel_average_rates = counts.astype(float) / (n_channels * bin_size)  # Hz
            channel_average_rates_list.append(channel_average_rates)

        trail_average_rates = np.mean(channel_average_rates_list, axis=0)
        return centers, trail_average_rates, channel_average_rates_list

    def _compute_ingroup_similarity(self, rates_list, visualize_time_bias, bin_size, similarity_duration):
        similarity = []
        start_index = int(visualize_time_bias/bin_size)
        duration = int(similarity_duration / bin_size)
        for i in range(len(rates_list)):
            for j in range(i + 1, len(rates_list)):
                a0 = rates_list[i][start_index:start_index+duration] - rates_list[i][start_index:start_index+duration].mean()
                b0 = rates_list[j][start_index:start_index+duration] - rates_list[j][start_index:start_index+duration].mean()
                denom = (np.linalg.norm(a0)*np.linalg.norm(b0)) + 1e-12
                corr = float(np.dot(a0, b0)/denom) if denom>0 else 0
                similarity.append(corr)
        return np.mean(similarity)
