import os
import matplotlib.pyplot as plt
import numpy as np

def plot_propagation_map(figure_save_path, session_name, data_idx, pattern_idx, time_arrival_list, max_firing_rate_list):
    result_path = os.path.join(figure_save_path, f"propagation_map/{session_name}/{pattern_idx}")
    os.makedirs(result_path, exist_ok=True)

    time_arrival_array = np.flipud(np.array(time_arrival_list).reshape(16, 8))
    max_firing_rate_array = np.flipud(np.array(max_firing_rate_list).reshape(16, 8))

    cmap1 = plt.cm.get_cmap('viridis').copy()
    cmap1.set_bad(color='0.7')

    cmap2 = plt.cm.get_cmap('viridis').copy()
    cmap2.set_bad(color='0.7')

    arr1 = np.ma.masked_invalid(time_arrival_array)
    arr2 = np.ma.masked_where(max_firing_rate_array == 0, max_firing_rate_array)

    fig, ax = plt.subplots(1, 2, figsize=(24, 12))

    im = ax[0].imshow(arr1, cmap=cmap1, origin='lower')
    fig.colorbar(im, ax=ax[0], label="Arrival Time (s)")
    ax[0].set_title(f"Propagation Delay - {session_name} (data {data_idx}, pattern {pattern_idx})")
    ax[0].set_xlabel("Channel (X-axis)")
    ax[0].set_ylabel("Channel (Y-axis)")

    im = ax[1].imshow(arr2, cmap=cmap2, origin='lower')
    fig.colorbar(im, ax=ax[1], label="Max Firing Rate (Hz)")
    ax[1].set_title(f"Max Firing Rate - {session_name} (data {data_idx}, pattern {pattern_idx})")
    ax[1].set_xlabel("Channel (X-axis)")
    ax[1].set_ylabel("Channel (Y-axis)")

    save_path = os.path.join(result_path, f"propagation_map_{data_idx}.png")
    fig.tight_layout()
    fig.savefig(save_path, format='png', dpi=300)
    plt.close(fig)