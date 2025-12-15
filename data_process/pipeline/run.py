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
from visualization.psth import plot_similarity

from config_file import configs


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
        
        self.silhouette_score_tsne = []
        self.silhouette_score_pca = []
        self.global_similarity_list = []


    def run(self):
        print("Loading data...")
        self.dataset_list = self.data_loader.load(self.spike_data_path)
        self.reorganize_channels()

        print("Data analysis...")
        for dataset in self.dataset_list:
            self.session_name = dataset.session_name
            self.spike_train = dataset.spike_train
            self.stimulation_init_time = dataset.stimulation_init_time

            if self.session_name.startswith("train"):
                self.num_patterns = configs.num_train_patterns
                self.pattern_dict = configs.train_pattern_dict
            else:
                self.num_patterns = configs.num_pretrain_patterns
                self.pattern_dict = configs.pretrain_pattern_dict


            self.compute_firing_rate(self.cfg["necessary_processes"]["firing_rate"])

            if "dimension_reduction" in self.cfg["optional_processes"] and self.session_name.startswith("train"):
                self.compute_dimension_reduction(self.cfg["optional_processes"]["dimension_reduction"])

            if "psth" in self.cfg["optional_processes"] and (self.session_name.startswith("train") or self.session_name.startswith("pretrain")):
                self.compute_psth(self.cfg["optional_processes"]["psth"])

            if "correlation_matrix" in self.cfg["optional_processes"] and self.session_name.startswith("spontaneous"):
                duration = self.cfg["optional_processes"]["correlation_matrix"]["duration"]
                plot_correlation_matrix(self.figure_save_path, self.session_name, self.firing_rate_result, self.step_size, self.num_channels, duration)

            if "propagation_map" in self.cfg["optional_processes"] and self.session_name.startswith("train"):\
                self.compute_propagation(self.cfg["optional_processes"]["propagation_map"])


        if "dimension_reduction" in self.cfg["optional_processes"]:
            plot_silhouette_score(self.figure_save_path, self.silhouette_score_tsne, self.silhouette_score_pca)
        
        if "psth" in self.cfg["optional_processes"]:
            plot_similarity(self.global_similarity_list, self.figure_save_path, self.num_patterns)

        return self
