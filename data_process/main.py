from pipeline.run import Pipeline

from config_file import configs

firing_rate_cfg = {
    "pre_offset": 0.5, 
    "post_offset": 0.02, 
    "window_size": 0.05, 
    "step_size": 0.01, 
    "gaussian_sigma": 0.01, 
    "method": "moving_average", 
    "plot_firing_rate": False
    }

dimension_reduction_cfg = {
    "execute": False,
    "method": ["pca"],
    "analysis_start_time": 0.02,
    "analysis_duration": 1,
}

psth_cfg = {
    "execute": False,
    "visualize_time_bias": 0.2,
    "visualize_duration": 1.2,
}

plotting_cfg = {
    "correlation_matrix": {"execute": False, "duration": 180},
    "propagation_map": {"execute": False, "rate_threshold": 20, "max_time_delay": 0.2},
}

pipeline_cfg = {
    "io": {
        "figure_save_path": configs.figure_save_path,
        "spike_data_path": configs.data_path,
        "mea_data_path": configs.mea_yaml_path,
        "excluded_channels": configs.excluded_channels,
        "reading_channels": configs.reading_channels,
    },
    "process": {
        "firing_rate": firing_rate_cfg,
        "dimension_reduction": dimension_reduction_cfg,
        "psth": psth_cfg,
        "additional_plotting": plotting_cfg
    },
}

pipe = Pipeline(pipeline_cfg).run()
