import os
import matplotlib.pyplot as plt
import numpy as np

def plot_propagation_map(figure_save_path, session_name, data_idx, pattern_idx, time_arrival_list):
    result_path = os.path.join(figure_save_path, f"propagation_map/{session_name}/{pattern_idx}")
    os.makedirs(result_path, exist_ok=True)

    time_arrival_array = np.flipud(np.array(time_arrival_list).reshape(16, 8))

    plt.figure(figsize=(10, 6))
    im = plt.imshow(time_arrival_array, cmap='viridis', origin='lower')
    plt.colorbar(im, label="Arrival Time (s)")
    plt.title(f"Propagation Map - {session_name} (data {data_idx}, pattern {pattern_idx})")
    plt.xlabel("Channel (X-axis)")
    plt.ylabel("Channel (Y-axis)")

    save_path = os.path.join(result_path, f"propagation_map_{data_idx}.png")
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()
