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

import neoml
import neoml.PythonWrapper as PythonWrapper
from neoml.MathEngine import MathEngine
from neoml.Blob import Blob
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------

def const(math_engine, shape, data):
    """Creates a blob of the specified shape filled with `data` value.
    """
    if not isinstance(math_engine, MathEngine):
        raise ValueError('The `math_engine` should be neoml.MathEngine.')

    np_shape = np.asarray(shape, dtype=np.int32)

    if len(np_shape) > 7:
        raise ValueError('The `shape` should have not more than 7 dimensions.')

    if np.isscalar(data):
        return Blob(PythonWrapper.blob_const(math_engine._internal, np_shape, float(data)))

    np_data = np.asarray(data, dtype=np.float32, order='C')

    if len(np_data.shape) > 7:
        raise ValueError('The `shape` should have not more than 7 dimensions.')

    return Blob(PythonWrapper.blob_const(math_engine._internal, np_shape, np_data))

def add(a, b):
    """Elementwise adds two blobs or a blob and a scalar value.
    """
    if not type(a) is Blob and not type(b) is Blob:
        raise ValueError('At least one of `a` and `b` should be neoml.Blob.')

    return a + b

def sub(a, b):
    """Elementwise subtracts two blobs or a blob and a scalar value.
    """
    if not type(a) is Blob and not type(b) is Blob:
        raise ValueError('At least one of `a` and `b` should be neoml.Blob.')

    return a - b

def mul(a, b):
    """Elementwise multiplies two blobs or a blob and a scalar value.
    """
    if not type(a) is Blob and not type(b) is Blob:
        raise ValueError('At least one of `a` and `b` should be neoml.Blob.')

    return a * b

def div(a, b):
    """Elementwise divides two blobs or a blob and a scalar value.
    """
    if not type(a) is Blob and not type(b) is Blob:
        raise ValueError('At least one of `a` and `b` should be neoml.Blob.')
    
    return a / b

def max(a, b):
    """Takes the elementwise maximum of a blob and a scalar value.
    """
    if type(a) is Blob:
        if a.size == 0:
            raise ValueError("The blob shouldn't be empty.")
        return Blob(PythonWrapper.blob_max(a._internal, float(b)))
    elif type(b) is Blob:
        if b.size == 0:
            raise ValueError("The blob shouldn't be empty.")
        return Blob(PythonWrapper.blob_max(float(a), b._internal))
    
    raise ValueError('At least one of `a` and `b` should be neoml.Blob.')

def sum(a, axes=None):
    """Calculates sum of blob elements along provided axes.
    If axes=None calculates the total sum.
    """
    if not type(a) is Blob:
        raise ValueError('`a` should be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob shouldn't be empty.")

    axes = np.array([] if axes is None else axes, dtype=np.int32)
    if not neoml.Utils.check_axes(axes):
        raise ValueError("`axes` should be unique and in range [0, 6].")

    return Blob(PythonWrapper.blob_sum(a._internal, axes))

def cumsum(a, axis=0):
    """Calculates cumulative sum of blob elements along provided axis.
    """
    if not type(a) is Blob:
        raise ValueError('`a` should be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob shouldn't be empty.")

    if not neoml.Utils.check_axes(axis):
        raise ValueError("`axis` should be in range [0, 6].")

    return Blob(PythonWrapper.blob_cumsum(a._internal, int(axis)))

def mean(a, axes=None):
    """Calculates mean of blob elements along provided axes.
    If axes=None calculates the total mean.
    """
    if not type(a) is Blob:
        raise ValueError('`a` should be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob shouldn't be empty.")

    axes = np.array([] if axes is None else axes, dtype=np.int32)
    if not neoml.Utils.check_axes(axes):
        raise ValueError("`axes` should be unique and in range [0, 6].")

    return Blob(PythonWrapper.blob_mean(a._internal, axes))

def neg(a):
    """Returns the negative of a blob or a number.
    """
    return -a;

def abs(a):
    """Takes absolute value of each blob element.
    """
    if not type(a) is Blob:
        raise ValueError('`a` should be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob shouldn't be empty.")
    
    return Blob(PythonWrapper.blob_abs(a._internal))

def log(a):
    """Takes the logarithm of each blob element.
    """
    if not type(a) is Blob:
        raise ValueError('`a` should be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob shouldn't be empty.")
    
    return Blob(PythonWrapper.blob_log(a._internal))

def exp(a):
    """Takes the exponential of each blob element.
    """
    if not type(a) is Blob:
        raise ValueError('`a` should be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob shouldn't be empty.")
    
    return Blob(PythonWrapper.blob_exp(a._internal))

def clip(blob, min_value, max_value):
    """Clips each element of the blob so that it fits between the specified limits.
    """
    if not type(blob) is Blob:
        raise ValueError('`blob` must be neoml.Blob.')

    if blob.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_clip(blob._internal, float(min_value), float(max_value)))

def concat(blobs, axis=0):
    """Merges the blobs along given axis.
    """
    if len(blobs) == 0:
        raise ValueError('Empty blobs.')

    if any([not type(blob) is Blob for blob in blobs]):
        raise ValueError('Every `blob` must be neoml.Blob.')

    if any([blob.size == 0 for blob in blobs]):
        raise ValueError("The blobs mustn't be empty.")

    return Blob(PythonWrapper.blob_concat([blob._internal for blob in blobs], int(axis)))

def broadcast(blob, shape):
    """Broadcast the blob shape.
    """
    if not type(blob) is Blob:
        raise ValueError('`blob` must be neoml.Blob.')

    if blob.size == 0:
        raise ValueError("The blobs mustn't be empty.")

    np_shape = np.asarray(shape, dtype=np.int32)

    if len(np_shape) > 7:
        raise ValueError('The `shape` should have not more than 7 dimensions.')

    for i, j in zip(blob.shape, [1] * (7 - len(np_shape)) + list(np_shape)):
        if i != j and i != 1:
            raise ValueError("The blobs have incompatible shapes.")

    return Blob(PythonWrapper.blob_broadcast(blob._internal, np_shape))

def reshape(blob, shape):
    """Reshape the blob.
    """
    if not type(blob) is Blob:
        raise ValueError('`blob` must be neoml.Blob.')

    if blob.size == 0:
        raise ValueError("The blobs mustn't be empty.")

    np_shape = np.asarray(shape, dtype=np.int32)

    if len(np_shape) > 7:
        raise ValueError('The `shape` should have not more than 7 dimensions.')

    if np.prod(np_shape) != blob.size:
        raise ValueError('`shape` is incompatible with current size.')

    PythonWrapper.blob_reshape(blob._internal, np_shape)

def pow(a, b):
    """Computes the power of one blob to another elementwise.
    """
    if not type(a) is Blob and not type(b) is Blob:
        raise ValueError('At least one of `a` and `b` should be neoml.Blob.')

    return a**b

def less(a, b):
    """Compare blobs elementwise.
    """
    if not type(a) is Blob and not type(b) is Blob:
        raise ValueError('At least one of `a` and `b` should be neoml.Blob.')

    return a < b

def top_k(a, k=1):
    """Finds values of the k largest elements in the blob.
    The result is a blob of size k.
    """
    if not type(a) is Blob:
        raise ValueError('`a` should be neoml.Blob.')

    if int(k) <= 0:
        raise ValueError('`k` should be > 0.')

    if a.size == 0:
        raise ValueError("The blob shouldn't be empty.")
    
    return Blob(PythonWrapper.blob_top_k(a._internal, int(k)))

def binary_cross_entropy(labels, preds, fromLogits):
    """Calculates binary cross-entropy for two blobs: the first one contains labels, the second one contains predictions.
    Blobs should be of the same shape.
    :math:`result = (1 - labels) * x + log(1 + exp(-x))`
    if `fromLogits` then `x = preds`, else :math:`x = log( clippedPreds / (1 - clippedPreds))`
    """
    if not type(labels) is Blob:
        raise ValueError('`labels` should be neoml.Blob.')

    if not type(preds) is Blob:
        raise ValueError('`preds` should be neoml.Blob.')

    if labels.shape != preds.shape:
        raise ValueError("The blobs should be of the same shape.")

    if labels.size == 0:
        raise ValueError("The blobs shouldn't be empty.")
    
    return Blob(PythonWrapper.blob_binary_cross_entropy(labels._internal, preds._internal, bool(fromLogits)))
