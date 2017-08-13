'''
Example usage of save_embeddings.py

Author: Liam Schoneveld
'''

import numpy as np
from save_embeddings import save_embeddings

N = 50 # number of items
D = 30 # dimensionality of vectors
H = 24 # height of images
W = 24 # width of images
D = 3   # depth of images

# Names for our two sets of feature vectors
name1 = 'testEmbedding1'
name2 = 'testEmbedding2'

# Images
images1 = np.random.uniform(size=[N, H, W, D])
images2 = np.random.uniform(size=[N * 2, H, W, D])

# Features
features1 = np.random.normal(size=[N, D])
features2 = np.random.uniform(size=[N * 2, D])

# Some example labels
labels1 = np.random.choice([i for i in 'ABCD'], size=N)
labels2 = np.random.choice([i for i in 'ABCD'], size=N * 2)

# Create dictionary
images_features_labels = {}
images_features_labels[name1] = [images1, features1, labels1]
images_features_labels[name2] = [images2, features2, labels2]

# Run function
save_embeddings(images_features_labels, './embeds')

# Then navigate to ./embeds and run: tensorboard --logdir=embeds
# Then head to http://localhost:6006/ and click Embeddings