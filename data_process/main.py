import os

from matplotlib import pyplot as plt
from pipeline.run import Pipeline


excluded_channels = ['A-000', 'A-003', 'A-028', 'A-029', 'A-030', 'B-010', 'B-019', 'B-022', 'B-031', 'C-003', 'C-006', 'C-007', 'C-020', 'C-022', 'D-016', 'D-017', 'D-018', 'D-019', 'D-020', 'D-021', 'D-022', 'D-023', 'D-025', 'D-026', 'D-030', 'A-015']

reading_channels = [
                    'A-000', 'A-001', 'A-002', 'A-003', 'A-004', 'A-005', 'A-006', 'A-007', 
                    'A-008', 'A-009', 'A-010', 'A-011', 'A-012', 'A-013', 'A-014', 'A-015', 
                    'A-016', 'A-017', 'A-018', 'A-019', 'A-020', 'A-021', 'A-022', 'A-023', 
                    'A-024', 'A-025', 'A-026', 'A-027', 'A-028', 'A-029', 'A-030', 'A-031',
                    'D-000', 'D-001', 'D-002', 'D-003', 'D-004', 'D-005', 'D-006', 'D-007', 
                    'D-008', 'D-009', 'D-010', 'D-011', 'D-012', 'D-013', 'D-014', 'D-015', 
                    'D-016', 'D-017', 'D-018', 'D-019', 'D-020', 'D-021', 'D-022', 'D-023', 
                    'D-024', 'D-025', 'D-026', 'D-027', 'D-028', 'D-029', 'D-030', 'D-031',
                    'B-000', 'B-001', 'B-002', 'B-003', 'B-004', 'B-005', 'B-006', 'B-007', 
                    'B-008', 'B-009', 'B-010', 'B-011', 'B-012', 'B-013', 'B-014', 'B-015', 
                    'B-016', 'B-017', 'B-018', 'B-019', 'B-020', 'B-021', 'B-022', 'B-023', 
                    'B-024', 'B-025', 'B-026', 'B-027', 'B-028', 'B-029', 'B-030', 'B-031',
                    'C-000', 'C-001', 'C-002', 'C-003', 'C-004', 'C-005', 'C-006', 'C-007', 
                    'C-008', 'C-009', 'C-010', 'C-011', 'C-012', 'C-013', 'C-014', 'C-015', 
                    'C-016', 'C-017', 'C-018', 'C-019', 'C-020', 'C-021', 'C-022', 'C-023', 
                    'C-024', 'C-025', 'C-026', 'C-027', 'C-028', 'C-029', 'C-030', 'C-031'
                    ]

mea_yaml_path = "/Users/aia/Desktop/cppcode/Intan-RHX/tcp/data_process/mea_topology/128_rhs.yaml"


data_path = "/Users/aia/Desktop/data/10_8_data/spike_train"
figure_save_path = "figures"

session_names = ["train1", "train2", "train3", "train4", "train5"]

score = []

firing_rate_cfg = {
    "pre_offset": 0, 
    "post_offset": 0, 
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
    "correlation_matrix": {"execute": True, "duration": 180},
    "silhouette_score": {"execute": True},
}

pipeline_cfg = {
    "io": {
        "figure_save_path": figure_save_path,
        "spike_data_path": data_path,
        "mea_data_path": mea_yaml_path,
        "excluded_channels": excluded_channels,
        "reading_channels": reading_channels,
    },
    "process": {
        "firing_rate": firing_rate_cfg,
        "dimension_reduction": dimension_reduction_cfg,
        "psth": psth_cfg,
        "additional_plotting": plotting_cfg
    },
}

pipe = Pipeline(pipeline_cfg).run()

# for phase in session_names:
#     spike_train_path = os.path.join(folder_path, f"spike_train{phase}.pkl")

#     if phase.startswith("train") or phase.startswith("pretrain"):
#         stimulation_init_time_path = os.path.join(folder_path, f"{phase}_stimulation_init_time.pkl")
#     else:
#         stimulation_init_time_path = None

    
#     pipline = (
#         DataProcess()
#         .load_data(spike_train_path, stimulation_init_time_path, session_name=phase)
#         .reorganize_channels(excluded_channels, reading_channels, file_path=mea_yaml_path, plot_layout=False)
#         .compute_firing_rate(pre_offset=0, post_offset=0, window_size=0.05, step_size=0.01, gaussian_sigma=0.01, method="moving_average", plot_firing_rate=False)
#         .data_analysis(method=["pca"], analysis_start_time=0.02, analysis_duration=1)
#         .plot_spike_train_with_psth(visualize_time_bias=0, visualize_duration=4)
#         .plot_correlation_matrix(duration=1800)
#         )
#     if phase.startswith("train"):
#         score.append(pipline.silhouette_score)

# plt.figure(figsize=(10, 5))
# plt.plot(range(len(score)), score, marker='o')
# plt.xticks(range(len(session_names)), session_names)
# plt.xlabel("Phase")
# plt.ylabel("Silhouette Score")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("silhouette_score.png", format="png", dpi=300)
# plt.close()
