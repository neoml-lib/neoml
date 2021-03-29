"""This sample generates dataset of 2-dimensional points
and performs clustering via K-Means with visualization of the result.
"""

__copyright__ = """

Copyright © 2017-2021 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

__license__ = 'Apache 2.0'

import numpy as np
import neoml
import matplotlib.pyplot as plt


# Data generation
# The data is generated as randomly chosen center + noise

n_dots = 128
n_clusters = 4
centers = np.array([(-2., -2.),
                   (-2., 2.),
                   (2., -2.),
                   (2., 2.)])
X = np.zeros(shape=(n_dots, 2), dtype=np.float32)
y = np.zeros(shape=(n_dots,), dtype=np.int32)
np.random.seed(123)
# Each point is center + N(0, 1)
for i in range(n_dots):
    cluster_id = np.random.randint(0, n_clusters)
    y[i] = cluster_id
    X[i, 0] = centers[cluster_id][0] + np.random.normal(0, 1)
    X[i, 1] = centers[cluster_id][1] + np.random.normal(0, 1)

# Clustering

kmeans = neoml.Clustering.KMeans(max_iteration_count=1000,
                                 cluster_count=n_clusters,
                                 thread_count=4)
labels, means, disps = kmeans.clusterize(X)

# Visualization

colors = {
    0: 'r',
    1: 'g',
    2: 'b',
    3: 'y'
}

# Create figure with 2 subplots
fig, axs = plt.subplots(ncols=2)
fig.set_size_inches(10, 5)

# Show ground truth
axs[0].set_title('Ground truth')
axs[0].scatter(X[:, 0], X[:, 1], marker='o', c=list(map(colors.get, y)))
axs[0].scatter(centers[:, 0], centers[:, 1], marker='x', c='black')

# Show NeoML markup
axs[1].set_title('K-Means')
axs[1].scatter(X[:, 0], X[:, 1], marker='o', c=list(map(colors.get, labels)))
axs[1].scatter(means[:, 0], means[:, 1], marker='x', c='black')

plt.show()
# As expected, outliers aren't labeled correctly
