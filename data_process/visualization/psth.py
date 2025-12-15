import os
import matplotlib.pyplot as plt


@staticmethod
def plot_spiketrain_raster(row, spike_train, ax, t_start, t_stop, visualize_time_bias, pattern_idx, pattern_dict): 
    spikes = spike_train.get_view(t_start, t_stop)

    ax.axvline(x=t_start + visualize_time_bias, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel("Channel")
    ax.eventplot(spikes)
    ax.text(0.1, 0.9, f"Index: {row}", transform=ax.transAxes, fontsize=10)

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
    similarity_list, 
    visualize_time_bias, 
    visualize_duration, 
    num_rows,
    bin_size,
    pattern_dict,
    repetition
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
            
            ax = plot_spiketrain_raster(row, spike_train, axes[row + 1], start_time, end_time, visualize_time_bias, pattern_idx, pattern_dict)
          
        ax = axes[0]
        ax.bar(centers[pattern_idx], rates[pattern_idx], width=bin_size, align='center')
        ax.text(0.5, 0.8, f"Similarity: {similarity_list[pattern_idx]:.3f}", transform=ax.transAxes, fontsize=10)
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

def plot_similarity(global_similarity_list, figure_save_path, num_patterns):
    similarity_path = os.path.join(figure_save_path, "PSTH")
    os.makedirs(similarity_path, exist_ok=True)

    sorted_global_similarity_list = sorted(global_similarity_list, key=lambda phase_idx: int(''.join(filter(str.isdigit, phase_idx[0]))))

    session_name_list = [session_name[0] for session_name in sorted_global_similarity_list]
    similarity_list = [data[1] for data in sorted_global_similarity_list]

    plt.figure(figsize=(10, 5))  
    for pattern_idx in range(num_patterns):
        plt.plot(session_name_list, [data[pattern_idx] for data in similarity_list], label=f"Pattern {pattern_idx}")

    plt.xlabel("Session")
    plt.ylabel("Similarity")
    plt.legend()
    plt.savefig(os.path.join(similarity_path, f"similarity.png"), format="png", dpi=300)
    plt.close("all")