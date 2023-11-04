import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE

def calculate_rv_score(cluster, global_variance):
    """
    Calculate the relative variance score for a cluster.
    """
    cluster_variance = np.var(cluster, axis=0)
    rv_score = (cluster_variance - global_variance) / np.std(global_variance)
    return rv_score

def vr_smote(data, n_clusters=5, high_rv_threshold=1.65, random_state=None):
    """
    Apply Variance Reduction SMOTE to the input data.
    """
    # Separate the features and the target
    X = data[:, :-1]
    y = data[:, -1]

    # Identify the minority class
    minority_class = np.argmin(np.bincount(y.astype(int)))
    minority_samples = X[y == minority_class]

    # Global variance of the minority class
    global_variance = np.var(minority_samples, axis=0)

    # Cluster the minority samples
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(minority_samples)

    # Calculate RV score for each cluster
    rv_scores = []
    for k in range(n_clusters):
        cluster = minority_samples[cluster_labels == k]
        rv_score = calculate_rv_score(cluster, global_variance)
        rv_scores.append(rv_score)

    # Initialize the arrays for high and medium variance clusters
    mh = np.array([]).reshape(0, X.shape[1])
    mm = np.array([]).reshape(0, X.shape[1])

    # Split clusters based on RV score
    for k in range(n_clusters):
        cluster = minority_samples[cluster_labels == k]
        if rv_scores[k] > high_rv_threshold:
            mh = np.vstack([mh, cluster])
        elif rv_scores[k] > -high_rv_threshold:
            mm = np.vstack([mm, cluster])

    # Prepare the final array for SMOTE
    m_array = np.vstack([mh, mm])

    # Apply SMOTE to high and medium variance clusters only
    smote = SMOTE(random_state=random_state)
    smote_data, smote_target = smote.fit_resample(m_array, np.array([minority_class]*m_array.shape[0]))

    # Combine the upsampled minority data with the original data
    smote_combined = np.hstack([smote_data, smote_target.reshape(-1, 1)])
    majority_data = data[y != minority_class]
    balanced_data = np.vstack([majority_data, smote_combined])

    return balanced_data

# Example usage:
# Assume `data` is a NumPy array with features and the last column as the target
# balanced_data = vr_smote(data)
