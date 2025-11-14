import os

import yaml
from matplotlib import pyplot as plt
from miv.core.datatype.spikestamps import Spikestamps
from visualization.mea_layout import plot_mea_layout

from config_file import configs

class OrganizeChannelsMixIn:
    def reorganize_channels(self):
        self.sorted_channel_data = self.load_mea_data(self.mea_data_path)
        plot_mea_layout(self.sorted_channel_data, self.figure_save_path)

        for dataset in self.dataset_list:
            if len(dataset.spike_train) != len(self.sorted_channel_data):
                raise ValueError("The number of channels in the dataset is not equal to the number of channels in the mea layout. Please check the mea layout file.")

            desired_sequence = [channel["idx"] for channel in self.sorted_channel_data]
            reading_channels = [idx for idx, channel in enumerate(self.sorted_channel_data) if f"{channel['chip_title']}-{channel['chip_index']}" in self.reading_channels]

            organized_spike_train = [dataset.spike_train[i] for i in desired_sequence]
            reading_spike_data = [organized_spike_train[i] for i in reading_channels]
            reading_spike_data = Spikestamps(reading_spike_data)
            dataset.spike_train = reading_spike_data
        
        return self


    def load_mea_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if configs.mea_type == "128_rhs":
            # 128 electrodes, 4 chips, 32 electrodes per chip
            chip_title = ["A", "B", "C", "D"]
            num_electrodes = [32, 32, 32, 32]
        
        if configs.mea_type == "512_long":
            chip_title = ["A", "B", "C", "D1", "D2"]
            num_electrodes = [128, 128, 128, 64, 64]

        channel_data = []
        chip_index = 0
        channel_index_in_group = 0
        for channel_index in range(sum(num_electrodes)):

            if channel_index % num_electrodes[chip_index] == 0 and channel_index != 0:
                chip_index += 1
                channel_index_in_group = 0

            channel_data.append({
                "chip_title": chip_title[chip_index],
                "chip_index": f"{channel_index_in_group:03d}",
                "channel_pos": data['pos'][channel_index],
                "idx": channel_index
            })
            channel_index_in_group += 1
            

        sorted_channel_data = sorted(sorted(channel_data, key=lambda x: x["channel_pos"][0]), key=lambda x: x["channel_pos"][1])

        return sorted_channel_data

