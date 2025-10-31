import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(figure_save_path, session_name, firing_rate_result, step_size, num_channels, duration):
    result_path = os.path.join(figure_save_path, f"correlation_matrix")
    os.makedirs(result_path, exist_ok=True)

    data_size = int(duration/step_size)
    spike_train = firing_rate_result["spike_rates"][:, -data_size:]

    for channel_idx in reversed(range(num_channels)):
        if np.sum(spike_train[channel_idx, :]) == 0:
            spike_train = np.delete(spike_train, channel_idx, axis=0)

    corr_matrix = np.corrcoef(spike_train)
    np.fill_diagonal(corr_matrix, 0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Neuron index')
    plt.ylabel('Neuron index')
    plt.savefig(os.path.join(result_path, f'{session_name}_correlation_matrix.png'), format='png', dpi=300)
    plt.close()
    