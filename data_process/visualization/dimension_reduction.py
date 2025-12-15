import os
import numpy as np
from matplotlib import pyplot as plt

def plot_pca_explained_variance(figure_save_path, session_name, pca):
    result_path = os.path.join(figure_save_path, f"PCA_explained_variance")
    os.makedirs(result_path, exist_ok=True)

    plt.figure(figsize=(6, 3))
    plt.bar(range(1, 11), pca.explained_variance_ratio_[:10] * 100, color='gray')
    plt.xlabel('PC number')
    plt.ylabel('% variance explained')
    plt.title('PCA Explained Variance')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f"{session_name}_explained_variance.png"), format='png', dpi=300)
    plt.close("all")


def plot_results_with_clusters(figure_save_path, session_name, results, num_patterns, test_name):
    result_path = os.path.join(figure_save_path, f"{test_name}_figures")
    os.makedirs(result_path, exist_ok=True)

    point_size = 30
    patterns_repo = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    pattern_colors_repo = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'black', 'yellow', 'cyan']

    patterns = patterns_repo[:num_patterns]
    pattern_colors = pattern_colors_repo[:num_patterns]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    num_points = results.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, num_points))
    scatter1 = ax1.scatter(results[:, 0], results[:, 1], c=colors, alpha=0.7, s=point_size)
    ax1.set_title(f'{test_name} - {session_name} (Time Sequence)')
    ax1.set_xlabel(f'{test_name} Component 1')
    ax1.set_ylabel(f'{test_name} Component 2')
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter1, ax=ax1)
    cbar.set_label('Time Sequence')
    
    for i in range(num_patterns):
        points_x = results[i::num_patterns, 0]
        points_y = results[i::num_patterns, 1]
        ax2.scatter(
            points_x,
            points_y,
            label=f"Pattern {patterns[i]}",
            alpha=0.8,
            s=point_size,
            color = pattern_colors[i]
        )

        for idx, (pos_x, pos_y) in enumerate(zip(points_x, points_y)):
            ax2.text(pos_x, 
                     pos_y, 
                     idx, 
                     fontsize=6, 
                     ha='center',
                     va='center')
    
    ax2.set_title(f'{test_name} - {session_name} (Feature-based Clusters)')
    ax2.set_xlabel(f'{test_name} Component 1')
    ax2.set_ylabel(f'{test_name} Component 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # ax2.set_ylim(-250, 250)
    # ax2.set_xlim(-1000, 0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'{session_name}_comparison.png'), format='png', dpi=300)
    plt.close()


def plot_silhouette_score(figure_save_path, silhouette_score_tsne, silhouette_score_pca):
    result_path = os.path.join(figure_save_path, f"silhouette_score")
    os.makedirs(result_path, exist_ok=True)
    
    if len(silhouette_score_tsne) != 0:
        sorted_silhouette_score_tsne = sorted(silhouette_score_tsne, key=lambda score: int(''.join(filter(str.isdigit, score[0]))))
        score_list = [score[1] for score in sorted_silhouette_score_tsne]
        label_list = [score[0] for score in sorted_silhouette_score_tsne]
        plt.figure(figsize=(10, 5))
        plt.plot(label_list, score_list, label='TSNE')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(result_path, f'tsne_silhouette_score.png'), format='png', dpi=300)
        plt.close()

    
    if len(silhouette_score_pca) != 0:
        sorted_silhouette_score_pca = sorted(silhouette_score_pca, key=lambda score: int(''.join(filter(str.isdigit, score[0]))))
        score_list = [score[1] for score in sorted_silhouette_score_pca]
        label_list = [score[0] for score in sorted_silhouette_score_pca]
        plt.figure(figsize=(10, 5))
        plt.plot(label_list, score_list, label='PCA')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(result_path, f'pca_silhouette_score.png'), format='png', dpi=300)
        plt.close()


