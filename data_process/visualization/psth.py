import os
import matplotlib.pyplot as plt


component = {
    0: [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27],
    1: [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31],
    2: [96, 97, 98, 99, 104, 105, 106, 107, 112, 113, 114, 115, 120, 121, 122, 123],
    3: [100, 101, 102, 103, 108, 109, 110, 111, 116, 117, 118, 119, 124, 125, 126, 127],
}

pattern_dict = {
    0: component[0]+component[1],
    1: component[0]+component[2],
    2: component[0]+component[3],
    3: component[1]+component[2],
    4: component[1]+component[3],
    5: component[2]+component[3],
}

@staticmethod
def plot_spiketrain_raster(spike_train, ax, t_start, t_stop, visualize_time_bias, pattern_idx, size=(30, 2)): 
    spikes = spike_train.get_view(t_start, t_stop)

    ax.axvline(x=t_start + visualize_time_bias, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel("Channel")
    ax.eventplot(spikes)
    
    if pattern_idx in pattern_dict:
        pattern_channels = pattern_dict[pattern_idx]
        for channel_idx in pattern_channels:
            ax.scatter(t_start+ visualize_time_bias-0.01, channel_idx, color='red', s=2, marker='o')
    
    ax.set_xlim(t_start, t_stop)
    return ax


def plot_spike_train_with_psth(
    spike_train,
    figure_save_path,
    session_name, 
    num_patterns, 
    start_time_list,
    end_time_list,
    centers, 
    rates, 
    similarity, 
    visualize_time_bias, 
    visualize_duration, 
    num_rows,
    bin_size
    ):

    firing_rate_path = os.path.join(figure_save_path, "PSTH")
    os.makedirs(firing_rate_path, exist_ok=True)
    phase_path = os.path.join(firing_rate_path, f"{session_name}")
    os.makedirs(phase_path, exist_ok=True)
    
    for pattern_idx in range(num_patterns):
        fig, axes = plt.subplots(num_rows + 1, 1, figsize=(30, 2*num_rows))

        for row in range(num_rows):
            start_time = start_time_list[row + pattern_idx * num_rows]
            end_time = end_time_list[row + pattern_idx * num_rows]
            
            ax = plot_spiketrain_raster(spike_train, axes[row + 1], start_time, end_time, visualize_time_bias, pattern_idx)
          
        ax = axes[0]
        ax.bar(centers[pattern_idx], rates[pattern_idx], width=bin_size, align='center')
        ax.text(0.5, 0.5, f"Similarity: {similarity:.3f}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean rate (Hz)")
        ax.set_title(f"Average PSTH", fontsize=10)
        ax.set_xlim(0, visualize_duration)

        plt.tight_layout()
        plt.savefig(
            os.path.join(phase_path, f"spiketrain_raster_for_pattern_{pattern_idx:01d}.png"),
            format="png",
            dpi=300,
        )
        plt.xlabel("Time (s)")
        plt.close("all")
