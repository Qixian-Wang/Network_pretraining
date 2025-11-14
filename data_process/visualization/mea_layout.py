import os

from matplotlib import pyplot as plt


def plot_mea_layout(sorted_channel_data, figure_save_path):
    x_length = max(channel["channel_pos"][0] for channel in sorted_channel_data)
    y_length = max(channel["channel_pos"][1] for channel in sorted_channel_data)

    plt.figure(figsize=(x_length/200, y_length/200))
    for channel in sorted_channel_data:
        plt.scatter(channel["channel_pos"][0], channel["channel_pos"][1], s=10, c='black')
        plt.text(channel["channel_pos"][0], channel["channel_pos"][1], 
                f"{channel['chip_title']}-{channel['chip_index']}, {channel['idx']}", fontsize=10)

    plt.savefig(os.path.join(figure_save_path, "layout.png"), format="png", dpi=300)
    plt.close('all')
