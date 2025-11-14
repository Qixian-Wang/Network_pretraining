import os

from analysismixin.moving_average_firing_rate import MovingAverageFiringRateMixIn
from analysismixin.dimension_reduction import DimensionReductionMixIn
from analysismixin.psth import PsthAnalysisMixin
from analysismixin.propagation import PropagationAnalysisMixin

from data_io.loader import DataLoader
from data_io.mea_layout import OrganizeChannelsMixIn

from visualization.correlation_matrix import plot_correlation_matrix
from visualization.dimension_reduction import plot_silhouette_score
from visualization.correlation_matrix import plot_correlation_matrix


class Pipeline(DimensionReductionMixIn, MovingAverageFiringRateMixIn, PsthAnalysisMixin, PropagationAnalysisMixin, OrganizeChannelsMixIn):

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_loader = DataLoader()

        self.spike_data_path = cfg["io"]["spike_data_path"]
        self.figure_save_path = cfg["io"]["figure_save_path"]
        self.mea_data_path = cfg["io"]["mea_data_path"]
        self.excluded_channels = cfg["io"]["excluded_channels"]
        self.reading_channels = cfg["io"]["reading_channels"]

        os.makedirs(self.figure_save_path, exist_ok=True)

        self.compute_firing_rate_flag = False
        self.dim_reduction_flag = False
        self.psth_flag = False
        self.correlation_matrix_flag = False
        self.propagation_map_flag = False

        if self.cfg["process"]["firing_rate"]["method"] is not None:
            self.compute_firing_rate_flag = True
            self.firing_rate_config = self.cfg["process"]["firing_rate"]

        if self.cfg["process"]["dimension_reduction"]["execute"]:
            self.dim_reduction_flag = True
            self.dim_reduction_config = self.cfg["process"]["dimension_reduction"]
            self.silhouette_score_tsne = []
            self.silhouette_score_pca = []

        if self.cfg["process"]["psth"]["execute"]:
            self.psth_flag = True
            self.psth_config = self.cfg["process"]["psth"]

        if self.cfg["process"]["additional_plotting"]["correlation_matrix"]["execute"]:
            self.correlation_matrix_flag = True
            self.duration = self.correlation_matrix_config["duration"]
            self.correlation_matrix_config = self.cfg["process"]["additional_plotting"]["correlation_matrix"]

        if self.cfg["process"]["additional_plotting"]["propagation_map"]["execute"]:
            self.propagation_map_flag = True
            self.propagation_map_config = self.cfg["process"]["additional_plotting"]["propagation_map"]


    def run(self):
        print("Loading data...")
        self.dataset_list = self.data_loader.load(self.spike_data_path)
        self.reorganize_channels()

        print("Data analysis...")
        for dataset in self.dataset_list:
            self.session_name = dataset.session_name
            self.spike_train = dataset.spike_train
            self.stimulation_init_time = dataset.stimulation_init_time

            if self.compute_firing_rate_flag:
                self.compute_firing_rate(self.firing_rate_config)

            if self.dim_reduction_flag and self.session_name.startswith("train"):
                self.compute_dimension_reduction(self.dim_reduction_config)
            
            if self.psth_flag and self.session_name.startswith("train"):
                self.compute_psth(self.psth_config)

            if self.correlation_matrix_flag and self.session_name.startswith("spontaneous"):
                plot_correlation_matrix(self.figure_save_path, self.session_name, self.firing_rate_result, self.step_size, self.num_channels, self.duration)

            if self.propagation_map_flag and self.session_name.startswith("train"):
                self.compute_propagation(self.propagation_map_config)
        
        
        if self.dim_reduction_flag:
            plot_silhouette_score(self.figure_save_path, self.silhouette_score_tsne, self.silhouette_score_pca)

        return self
