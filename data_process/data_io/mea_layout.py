import os

import yaml
from matplotlib import pyplot as plt
from miv.core.datatype.spikestamps import Spikestamps


class OrganizeChannelsMixIn:
    @staticmethod
    def _bank_encode(channel_name):
        bank = channel_name[0].upper()
        idx = int(channel_name[-3:])
        base = {"A": 0, "B": 32, "C": 64, "D": 96}[bank]
        return base + idx


    def reorganize_channels(self):
        self.sorted_channel_data = self.load_mea_data(self.mea_data_path)

        for dataset in self.dataset_list:
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

        # 128 electrodes, 4 chips, 32 electrodes per chip
        chip_title = ["A", "B", "C", "D"]
        num_electrodes = 128

        channel_data = []
        chip_index = 0
        channel_index_in_group = 0
        for channel_index in range(num_electrodes):

            if channel_index % 32 == 0 and channel_index != 0:
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


    def _plot_mea_layout(self):
        plt.figure(figsize=(10, 10))
        for channel in self.sorted_channel_data:
            plt.scatter(channel["channel_pos"][0], channel["channel_pos"][1], s=10, c='black')
            plt.text(channel["channel_pos"][0], channel["channel_pos"][1], 
                    f"{channel['chip_title']}-{channel['chip_index']}, {channel['idx']}", fontsize=10)

        plt.savefig(os.path.join(self.figure_save_path, "layout.png"), format="png", dpi=300)
        plt.close('all')
