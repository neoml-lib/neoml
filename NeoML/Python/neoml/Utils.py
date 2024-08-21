""" Copyright (c) 2017-2024 ABBYY

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from neoml.Dnn.Dnn import Layer


def check_input_layers(input_layers, layer_count):
    min_count = 0
    max_count = 0
    if isinstance(layer_count, int):
        if layer_count == 0:
            min_count = 1
            max_count = 9223372036854775807
        else:
            min_count = int(layer_count)
            max_count = int(layer_count)
    else:
        min_count = int(layer_count[0])
        max_count = int(layer_count[1])

    layers = []
    outputs = []

    # helper functions to detect if an object is a link to another layer and add this link
    _is_link = lambda link: isinstance(link, Layer) \
        or (len(link) == 2 and isinstance(link[0], Layer) and isinstance(link[1], int))

    def _add_link(link):
        internal_layer, index = (link._internal, 0) if isinstance(link, Layer) \
            else (link[0]._internal, int(link[1]))
        if index < 0:
            raise ValueError('The indices in the `input_layers` must be non-negative')
        layers.append(internal_layer)
        outputs.append(index)

    if _is_link(input_layers):
        _add_link(input_layers)
    else:
        if len(input_layers) == 0:
            raise ValueError('The `input_layers` must contain at least one layer.')
        for input_layer in input_layers:
            if not _is_link(input_layer):
                raise ValueEror('Each entry of `input_layers` must be Layer or tuple(Layer, int)')
            _add_link(input_layer)

    if len(layers) < min_count or len(layers) > max_count:
        raise ValueError('The layer has (' + str(min_count) + ', ' + str(max_count) + ') inputs.')
    return layers, outputs

def convert_data(X):
    if issparse(X):
        return csr_matrix(X, dtype=np.float32)

    data = np.asarray(X, dtype=np.float32, order='C')
    if data.ndim != 2:
        raise ValueError('X must be of shape (n_samples, n_features)')
    return data

def get_data(X):
    if issparse(X):
        return X.indices, X.data, X.indptr, True
    height, width = X.shape
    indptr = np.array([i * width for i in range(height+1)], dtype=np.int32, order='C')
    return np.array([]), X.ravel(), indptr, False

def check_can_broadcast(X, Y):
    for i, j in zip(X.shape, Y.shape):
        if i != j and i != 1 and j != 1:
            return False
    return True

def check_axes(axes):
    axes = np.array(axes)
    all_unique = np.all(np.unique(axes, return_counts=True)[1] == 1)
    return all_unique and np.all((axes >= 0) * (axes < 7))
