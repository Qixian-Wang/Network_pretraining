import numpy as np

from visualization.psth import plot_spike_train_with_psth

from config_file import configs


class PsthAnalysisMixin:
    def compute_psth(self, config):
        visualize_time_bias = config["visualize_time_bias"]
        visualize_duration = config["visualize_duration"]

        num_rows = len(self.stimulation_init_time) // configs.num_patterns
        bin_size = 0.001

        start_time_list = []
        end_time_list = []
        centers_list = []
        rates_list = []

        for pattern_idx in range(configs.num_patterns):
            aligned_data = []

            for row in range(num_rows):
                start_time = self.stimulation_init_time[row * configs.num_patterns + pattern_idx] - visualize_time_bias
                end_time = start_time + visualize_duration

                start_time_list.append(start_time)
                end_time_list.append(end_time)

                aligned_data.append(self._collect_window_spikes(start_time, end_time))
            
            centers, rates, rates_list = self._compute_psth(aligned_data, window_len=visualize_duration, n_channels=len(self.spike_train), bin_size=bin_size)

            centers_list.append(centers)
            rates_list.append(rates)

            similarity = self._compute_ingroup_similarity(rates_list, visualize_time_bias, bin_size)


        plot_spike_train_with_psth(
            spike_train=self.spike_train,
            figure_save_path=self.figure_save_path,
            session_name=self.session_name,
            num_patterns=configs.num_patterns,
            start_time_list=start_time_list,
            end_time_list=end_time_list,
            centers=centers_list,
            rates=rates_list,
            similarity=similarity,
            visualize_time_bias=visualize_time_bias,
            visualize_duration=visualize_duration,
            num_rows=num_rows,
            bin_size=bin_size
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
        rates_list = []

        for data in aligned_data:
            counts, _ = np.histogram(data, bins=edges)
            rates = counts.astype(float) / (n_channels * bin_size)  # Hz
            rates_list.append(rates)

        rates = np.mean(rates_list, axis=0)
        return centers, rates, rates_list

    def _compute_ingroup_similarity(self, rates_list, visualize_time_bias, bin_size):
        similarity = []
        start_index = int(visualize_time_bias/bin_size)
        for i in range(len(rates_list)):
            for j in range(i + 1, len(rates_list)):
                a0 = rates_list[i][start_index:start_index+200] - rates_list[i][start_index:start_index+200].mean()
                b0 = rates_list[j][start_index:start_index+200] - rates_list[j][start_index:start_index+200].mean()
                denom = (np.linalg.norm(a0)*np.linalg.norm(b0)) + 1e-12
                corr = float(np.dot(a0, b0)/denom) if denom>0 else 0
                similarity.append(corr)
        return np.mean(similarity)
