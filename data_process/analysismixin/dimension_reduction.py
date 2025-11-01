import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from visualization.dimension_reduction import plot_pca_explained_variance, plot_results_with_clusters
from analysismixin.truncate_data import TruncateDataMixIn

from config_file import configs


class DimensionReductionMixIn(TruncateDataMixIn):

    def compute_dimension_reduction(self, config):
        method = config["method"]
        analysis_start_time = config["analysis_start_time"]
        analysis_duration = config["analysis_duration"]

        if self.stimulation_init_time is None:
            return self

        data = self.firing_rate_result["spike_rates"]

        truncated_data = self.truncate_data(
            data,
            analysis_start_time,
            analysis_duration,
        )

        self.organized_data = truncated_data.reshape(truncated_data.shape[0], -1)

        num_stim = truncated_data.shape[0]
        self.labels = np.tile(np.arange(configs.num_patterns), num_stim // configs.num_patterns)

        if "tsne" in method:
            silhouette_score = self.tsne_analysis()
            self.silhouette_score_tsne.append([self.session_name, silhouette_score])
        if "pca" in method:
            silhouette_score = self.pca_analysis()
            self.silhouette_score_pca.append([self.session_name, silhouette_score])

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
        plot_results_with_clusters(self.figure_save_path, self.session_name, X_pca, configs.num_patterns, test_name="PCA")

        return silhouette
