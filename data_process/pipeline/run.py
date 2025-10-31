from analysismixin.moving_average_firing_rate import MovingAverageFiringRateMixIn
from analysismixin.dimension_reduction import DimensionReductionMixIn
from analysismixin.psth import PsthAnalysisMixin
from visualization.correlation_matrix import plot_correlation_matrix
from data_io.mea_layout import OrganizeChannelsMixIn


from data_io.loader import DataLoader


class Pipeline(DimensionReductionMixIn, MovingAverageFiringRateMixIn, PsthAnalysisMixin, OrganizeChannelsMixIn):

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_loader = DataLoader()

        self.spike_data_path = cfg["io"]["spike_data_path"]
        self.figure_save_path = cfg["io"]["figure_save_path"]
        self.mea_data_path = cfg["io"]["mea_data_path"]
        self.excluded_channels = cfg["io"]["excluded_channels"]
        self.reading_channels = cfg["io"]["reading_channels"]

    def run(self):
        print("Loading data...")
        self.dataset_list = self.data_loader.load(self.spike_data_path)
        self.reorganize_channels()

        for dataset in self.dataset_list:
            self.session_name = dataset.session_name
            self.spike_train = dataset.spike_train
            self.stimulation_init_time = dataset.stimulation_init_time

            if self.session_name.startswith("train"):
                self.num_patterns = 6

            elif self.session_name.startswith("pretrain"):
                self.num_patterns = 4

            print("Computing firing rate...")
            if self.cfg["process"]["firing_rate"]["method"] is not None:
                config = self.cfg["process"]["firing_rate"]
                self.compute_firing_rate(config)

            print("Data analysis...")
            if self.cfg["process"]["dimension_reduction"]["execute"]:
                config = self.cfg["process"]["dimension_reduction"]
                self.compute_dimension_reduction(config)
            
            if self.cfg["process"]["psth"]["execute"]:
                config = self.cfg["process"]["psth"]
                self.compute_psth(config)

            if self.cfg["process"]["additional_plotting"]["correlation_matrix"]["execute"] and self.session_name.startswith("spontaneous"):
                duration = self.cfg["process"]["additional_plotting"]["correlation_matrix"]["duration"]
                plot_correlation_matrix(self.figure_save_path, self.session_name, self.firing_rate_result, self.step_size, self.num_channels, duration)

            self.plot_silhouette_score()
            
        return self
