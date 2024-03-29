# -*- coding: utf-8 -*-
"""anime_face_cluster.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1at9w7rcCUqAwHiTQiq3DJ0E04VjWTNBA
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# %matplotlib inline

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive

drive.mount("/content/gdrive/", force_remount=True)
# %cd gdrive/MyDrive/

try:
    images_flat = np.load(
        "anime_face_images/anime_face_images.npy",
        allow_pickle=True
    )
except:
    import os
    # load the images
    images = []
    for i in os.listdir("anime_face_images/"):
      images.append(mpimg.imread(f"anime_face_images/{i}"))

    # extract features from images
    images_flat = np.asarray([i.flatten().reshape(64, 64, 3) for i in images])
    np.save("anime_face_images/anime_face_images.npy", images_flat)

# normalize the features
normalized_features = (images_flat - np.min(images_flat)) / \
    (np.max(images_flat) - np.min(images_flat))

normalized_features = np.reshape(
    normalized_features,
    (
        normalized_features.shape[0],
        normalized_features.shape[1] *
        normalized_features.shape[2] * normalized_features.shape[3]
    )
)

# Perform PCA to reduce the dimensionality of the data
pca = PCA(n_components=1000)
reduced_features = pca.fit_transform(normalized_features)

# apply KMeans clustering on the reduced features
kmeans = KMeans(n_clusters=6, random_state=0).fit(reduced_features)

np.save("anime_face_images/kmeans_labels.npy", kmeans.labels_)