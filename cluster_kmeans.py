import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.preprocessing
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Load our cleaned data
data_dir = 'data'
input_file = 'processed_reddit_data'
df = pd.read_csv(os.path.join(data_dir, input_file), sep=r'\|\*\|', on_bad_lines='warn', encoding='utf-8')
tweets = df['processed_strings'].tolist()
tweets = [item for sublist in tweets for item in eval(sublist)]
long = ' '.join(tweets)
print("number of data points: ", len(tweets))
#vectorize the data
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(long)

## K-Means Clustering
kmeans_model = KMeans(n_clusters=50, n_init=20)
X_norm = sklearn.preprocessing.normalize(X_tfidf, axis=1)
kmeans_model.fit(X_norm)
labels = kmeans_model.labels_
cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
print(f"Number of elements asigned to each cluster: {cluster_sizes}")

distances = np.zeros(X_norm.shape[0])
cluster_centers = kmeans_model.cluster_centers_
for i in range(X_norm.shape[0]):
    label = labels[i]
    centroid = cluster_centers[label]
    distance = np.linalg.norm(X_norm[i] - centroid)
    distances[i] = distance
average_distance = np.mean(distances)
print("Average intra-cluster distance:", average_distance)
inter_cluster_distances = pairwise_distances(cluster_centers).flatten()
average_inter_cluster_distance = np.sum(inter_cluster_distances) / (len(cluster_centers)**2 - len(cluster_centers))
print("Average inter-cluster distance:", average_inter_cluster_distance)
# silhouette_avg = silhouette_score(X_norm, labels)
# print("Silhouette Score:", silhouette_avg)


# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42, init='random', metric="l2")
X_tsne = tsne.fit_transform(X_norm)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels)
plt.title("t-SNE Visualization of Clustering Results")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()

# map the labels back to the dataframe and save it
list_df = pd.DataFrame({'comment':tweets, 'label':labels})
header = '|*|'.join(list_df.columns)
print(header)
np.savetxt(os.path.join(data_dir, "labeled_" + str(input_file)), list_df, fmt=["%s", '%d'], delimiter="|*|", header=header, encoding='utf-8', comments='')
print("Done")
