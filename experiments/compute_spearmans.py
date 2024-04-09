import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Metrics for each run
metrics = np.array([
    [0.79, 0.69, 0.53, 0.84, 0.54, 0.99, 0.23, 0.56, 0.61, 0.56, 0.99, 0.98],
    [0.42, 0.40, 0.33, 0.77, 0.52, 0.99, 0.277, 0.36, 0.544, 0.42, 0.97, 0.84],
    [0.68, 0.61, 0.30, 0.67, 0.47, 0.98, 0.259, 0.33, 0.477, 0.31, 0.95, 0.82],

])

# Index labels
index_labels = ['DCI Mod', 'DCI Comp', 'DCI Expl', 'EDI Mod', 'EDI Comp', 'EDI Expl',
                'MIG', 'SAP', 'DCIMIG', 'MIG-sup', 'Modularity', 'Z_min Variance']

# Compute Spearman's correlation coefficient for each pair of metrics
correlation_matrix = np.zeros((metrics.shape[1], metrics.shape[1]))
for i in range(metrics.shape[1]):
    for j in range(metrics.shape[1]):
        correlation, _ = spearmanr(metrics[:, i], metrics[:, j])
        correlation_matrix[i, j] = correlation

# Plot heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f",
            xticklabels=index_labels,
            yticklabels=index_labels)
plt.title("Spearman's Rank Correlation Coefficients")
plt.xlabel("Metrics")
plt.ylabel("Metrics")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
