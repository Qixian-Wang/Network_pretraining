from tkinter.constants import FALSE
from pipeline.run import Pipeline

from config_file import configs

firing_rate_cfg = {
    "pre_offset": 0.05, 
    "post_offset": 0.02, 
    "window_size": 0.05, 
    "step_size": 0.01, 
    "gaussian_sigma": 0.01, 
    "method": "moving_average", 
    "plot_firing_rate": False
    }

dimension_reduction_cfg = {
    "method": ["pca"],
    "analysis_start_time": 0.02,
    "analysis_duration": 1,
}

psth_cfg = {
    "visualize_time_bias": 0.2,
    "visualize_duration": 1.2,
    "bin_size": 0.001,
}

pipeline_cfg = {
    "io": {
        "figure_save_path": configs.figure_save_path,
        "spike_data_path": configs.data_path,
        "mea_data_path": configs.mea_yaml_path,
        "excluded_channels": configs.excluded_channels,
        "reading_channels": configs.reading_channels,
    },

    "necessary_processes": {
        "firing_rate": firing_rate_cfg,
    },

    "optional_processes": {
        "dimension_reduction": dimension_reduction_cfg,
        "psth": psth_cfg,
        "correlation_matrix": {"duration": 180},
        "propagation_map": {"rate_threshold": 20, "max_time_delay": 0.2},
    }
}

pipe = Pipeline(pipeline_cfg).run()
