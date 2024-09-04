"""The sample compares different boosting algorithms
on 20newsgroups dataset
"""

__copyright__ = """

Copyright © 2017-2024 ABBYY

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

import neoml
import numpy as np
import time

# Get data
from sklearn.datasets import fetch_20newsgroups_vectorized

train_data = fetch_20newsgroups_vectorized(subset='train')
test_data = fetch_20newsgroups_vectorized(subset='test')


def accuracy(model, X, y):
    """Returns the accuracy of model on the given data"""
    correct = sum(1 for label, probs in zip(y, model.classify(X))
                  if label == np.argmax(probs))
    return float(correct)/len(y)


# These arguments will be used for every builder_type
shared_kwargs = {
    'loss' : 'binomial',
    'iteration_count' : 100,
    'learning_rate' : 0.1,
    'subsample' : 1.,
    'subfeature' : 0.25,
    'random_seed' : 1234,
    'max_depth' : 6,
    'max_node_count' : -1,
    'l1_reg' : 0.,
    'l2_reg' : 1.,
    'prune' : 0.,
    'thread_count' : 1,
}


# Train and test boosting for every builder type
for builder in ['full', 'hist', 'multi_full']:
    start = time.time()
    boost_kwargs = { **shared_kwargs, 'builder_type' : builder}
    classifier = neoml.GradientBoost.GradientBoostClassifier(**boost_kwargs)
    model = classifier.train(train_data.data, train_data.target)
    run_time = time.time() - start
    acc = accuracy(model, test_data.data, test_data.target)
    print(f'{builder}  Accuracy: {acc:.4f}  Time: {run_time:.2f} sec.')
