import os
from matplotlib import pyplot as plt
import numpy as np

def plot_phase_firing_rate(figure_save_path, session_name, spike_train, firing_rate_result, step_size, image_time=10):
    end_time = spike_train.get_last_spikestamp()
    num_channels = len(spike_train)

    firing_rate_path = os.path.join(figure_save_path, "firing_rate")
    os.makedirs(firing_rate_path, exist_ok=True)
        
    spike_rates_matrix = firing_rate_result['spike_rates']
    window_centers = firing_rate_result['window_centers']

    phase_path = os.path.join(firing_rate_path, f"{session_name}")
    os.makedirs(phase_path, exist_ok=True)
        
    num_images = int(np.ceil(end_time / image_time))
        
    for image_idx in range(num_images):
        image_start_time = image_time * image_idx
        actual_image_duration = min(image_time, end_time - image_idx * image_time)

        windows_per_step = int(1 / step_size) if step_size > 0 else 1
        num_windows_per_image = int(actual_image_duration * windows_per_step)
        start_window = int(image_idx * image_time * windows_per_step)
        end_window = min(start_window + num_windows_per_image, len(window_centers))
        
        t_slice = window_centers[start_window:end_window]
        spike_data_slice = spike_rates_matrix[:, start_window:end_window]
        
        plt.figure(figsize=(12, 8))
        for ch_idx in range(num_channels):
            plt.plot(t_slice, spike_data_slice[ch_idx, :], 
                    linewidth=2, color='orange', alpha=0.8)
            plt.ylabel("Firing rate (Hz)", fontsize=12)
            plt.xlabel("Time (s)", fontsize=12)
            plt.title(f"Firing rate from {image_start_time} to {image_start_time + actual_image_duration}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
                
        save_path = os.path.join(phase_path, f"moving_average_firing_rate_{image_idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.close('all')