import numpy as np

class TruncateDataMixIn:
    def truncate_data(self, data, analysis_start_time, analysis_duration):
        num_stimulation = len(self.stimulation_init_time)
        starting_data_size = int(analysis_start_time / self.step_size)
        truncated_data_size = int(analysis_duration / self.step_size)
        complete_cycles = num_stimulation // self.num_patterns

        print(f"Stimulation number: {num_stimulation} stimulations; patterns: {self.num_patterns}; complete_cycles: {complete_cycles}", flush=True)
        
        truncated_data_analyze = []
        for stimulation_idx in range(num_stimulation):
            start_idx = int(self.stimulation_init_time[stimulation_idx] / self.step_size) + starting_data_size
            end_idx = start_idx + truncated_data_size

            if end_idx <= data.shape[1]:
                truncated_data_analyze.append(data[:, start_idx:end_idx])
            else:
                raise ValueError("analysis duration maybe too large")

        truncated_data = np.array(truncated_data_analyze)
        print(f"Final truncated data shape: {truncated_data.shape}", flush=True)

        return truncated_data