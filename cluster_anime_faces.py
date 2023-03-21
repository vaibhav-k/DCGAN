from sklearn.decomposition import PCA
import cv2
import numpy as np
from sklearn.cluster import KMeans

# load the images
imgs = np.load("./keras_models/merged_data.npy",
               allow_pickle=True, encoding='bytes')

# extract features from images
# imgs_flat = np.asarray([i.flatten() for i in imgs])
imgs_flat = np.asarray([i.flatten().reshape(64, 64, 3) for i in imgs])

# normalize the features
normalized_features = (imgs_flat - np.min(imgs_flat)) / \
    (np.max(imgs_flat) - np.min(imgs_flat))

normalized_features = np.reshape(
    normalized_features,
    (
        normalized_features.shape[0],
        normalized_features.shape[1] *
        normalized_features.shape[2] * normalized_features.shape[3]
    )
)


# Perform PCA to reduce the dimensionality of the data
pca = PCA(n_components=100)
reduced_features = pca.fit_transform(normalized_features)

# Apply KMeans clustering on the reduced features
kmeans = KMeans(n_clusters=6, random_state=0).fit(reduced_features)

# visualize the clusters
cluster_labels = kmeans.labels_
for i, img in enumerate([img1, img2, img3]):
    cv2.imshow('Cluster ' + str(cluster_labels[i]), img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save the cluster labels
with open("./keras_models/kmeans_labels.txt") as f:
    f.write(cluster_labels)
