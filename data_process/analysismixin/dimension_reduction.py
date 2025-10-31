import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from visualization.dimension_reduction import plot_pca_explained_variance, plot_results_with_clusters


class DimensionReductionMixIn:

    def compute_dimension_reduction(self, config):
        method = config["method"]
        analysis_start_time = config["analysis_start_time"]
        analysis_duration = config["analysis_duration"]

        if self.stimulation_init_time is None:
            return self

        data = self.firing_rate_result["spike_rates"]

        self._truncate_data(
            data,
            analysis_start_time,
            analysis_duration,
        )

        self.organized_data = self.result_analyze.reshape(self.result_analyze.shape[0], -1)

        num_stim = self.result_analyze.shape[0]
        self.labels = np.tile(np.arange(self.num_patterns), num_stim // self.num_patterns)

        if "tsne" in method:
            self.tsne_analysis()
        if "pca" in method:
            self.silhouette_score = self.pca_analysis()

        return self


    def tsne_analysis(self):  
        tsne = TSNE(
            n_components=2,
            perplexity=10,
            max_iter=2000,
            learning_rate='auto',
            random_state=0,
            verbose=1
        )

        X_tsne = tsne.fit_transform(self.organized_data)
        silhouette = silhouette_score(X_tsne, self.labels)
        print(f"TSNE Silhouette Score for {self.session_name}: {silhouette:.3f}", flush=True)

        self._plot_results_with_clusters(X_tsne, test_name="TSNE")

        return silhouette


    def pca_analysis(self):
        pca = PCA(n_components=10)
        X_pca = pca.fit_transform(self.organized_data)
        silhouette = silhouette_score(X_pca, self.labels)
        print(f"PCA Silhouette Score for {self.session_name}: {silhouette:.3f}", flush=True)
 
        plot_pca_explained_variance(self.figure_save_path, self.session_name, pca)
        plot_results_with_clusters(self.figure_save_path, self.session_name, X_pca, self.num_patterns, test_name="PCA")

        return silhouette


    def _truncate_data(self, data, analysis_start_time, analysis_duration):
        num_stimulation = len(self.stimulation_init_time)
        starting_data_size = int(analysis_start_time / self.step_size)
        truncated_data_size = int(analysis_duration / self.step_size)
        complete_cycles = num_stimulation // self.num_patterns

        print(f"Stimulation number: {num_stimulation} stimulations; patterns: {self.num_patterns}; complete_cycles: {complete_cycles}", flush=True)
        
        truncated_data_analyze = []
        for stimulation_idx in range(num_stimulation):
            start_idx = int(self.stimulation_init_time[stimulation_idx] / self.step_size) + starting_data_size
            end_idx = start_idx + truncated_data_size

            if end_idx <= data.shape[1]:
                truncated_data_analyze.append(data[:, start_idx:end_idx])
            else:
                raise ValueError("analysis duration maybe too large")

        self.result_analyze = np.array(truncated_data_analyze)
        print(f"Final truncated data shape: {self.result_analyze.shape}", flush=True)
        