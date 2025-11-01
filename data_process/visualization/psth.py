import os
import matplotlib.pyplot as plt

from config_file import configs


@staticmethod
def plot_spiketrain_raster(spike_train, ax, t_start, t_stop, visualize_time_bias, pattern_idx): 
    spikes = spike_train.get_view(t_start, t_stop)

    ax.axvline(x=t_start + visualize_time_bias, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel("Channel")
    ax.eventplot(spikes)
    
    if pattern_idx in configs.pattern_dict:
        pattern_channels = configs.pattern_dict[pattern_idx]
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
