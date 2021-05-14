""" Copyright (c) 2017-2020 ABBYY Production LLC

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

import neoml.PythonWrapper as PythonWrapper
from neoml.Blob import Blob

# ----------------------------------------------------------------------------------------------------------------------

def const(math_engine, shape, value):
    """
    """
    if not isinstance(math_engine, MathEngine.MathEngine):
        raise ValueError('The `math_engine` must be neoml.MathEngine.')

    np_shape = numpy.array(shape, dtype=numpy.int32, copy=False)

    if len(np_shape.shape) > 7:
        raise ValueError('The `shape` must have not more then 7 dimensions.')
    
    return Blob(PythonWrapper.blob_const(mathEngine._internal, np_shape, float(value)))

def const(math_engine, shape, data):
    """
    """
    if not isinstance(math_engine, MathEngine.MathEngine):
        raise ValueError('The `math_engine` must be neoml.MathEngine.')

    np_shape = numpy.array(shape, dtype=numpy.int32, copy=False)

    if len(np_shape.shape) > 7:
        raise ValueError('The `shape` must have not more then 7 dimensions.')

    np_data = numpy.array(data, copy=False, order='C')

    if len(np_data.shape) > 7:
        raise ValueError('The `shape` must have not more then 7 dimensions.')
    
    return Blob(PythonWrapper.blob_const(mathEngine._internal, np_shape, np_data))

def add(a, b):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if not type(b) is Blob:
        raise ValueError('`b` must be neoml.Blob.')

    if a.shape != b.shape:
        raise ValueError("The blobs must have the same shape.")

    if a.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_add(a._internal, b._internal))

def add(a, b):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob mustn't be empty.")
    
    return Blob(PythonWrapper.blob_add(a._internal, float(b)))

def add(a, b):
    """
    """
    if not type(b) is Blob:
        raise ValueError('`b` must be neoml.Blob.')

    if b.size == 0:
        raise ValueError("The blob mustn't be empty.")
    
    return Blob(PythonWrapper.blob_add(float(a), b._internal))

def sub(a, b):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if not type(b) is Blob:
        raise ValueError('`b` must be neoml.Blob.')

    if a.shape != b.shape:
        raise ValueError("The blobs must have the same shape.")

    if a.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_sub(a._internal, b._internal))

def sub(a, b):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob mustn't be empty.")
    
    return Blob(PythonWrapper.blob_sub(a._internal, float(b)))

def sub(a, b):
    """
    """
    if not type(b) is Blob:
        raise ValueError('`b` must be neoml.Blob.')

    if b.size == 0:
        raise ValueError("The blob mustn't be empty.")
    
    return Blob(PythonWrapper.blob_sub(float(a), b._internal))

def mult(a, b):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if not type(b) is Blob:
        raise ValueError('`b` must be neoml.Blob.')

    if a.shape != b.shape:
        raise ValueError("The blobs must have the same shape.")

    if a.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_mult(a._internal, b._internal))

def mult(a, b):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob mustn't be empty.")
    
    return Blob(PythonWrapper.blob_mult(a._internal, float(b)))

def mult(a, b):
    """
    """
    if not type(b) is Blob:
        raise ValueError('`b` must be neoml.Blob.')

    if b.size == 0:
        raise ValueError("The blob mustn't be empty.")
    
    return Blob(PythonWrapper.blob_mult(float(a), b._internal))

def div(a, b):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if not type(b) is Blob:
        raise ValueError('`b` must be neoml.Blob.')

    if a.shape != b.shape:
        raise ValueError("The blobs must have the same shape.")

    if a.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_div(a._internal, b._internal))

def div(a, b):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob mustn't be empty.")
    
    return Blob(PythonWrapper.blob_div(a._internal, float(b)))

def div(a, b):
    """
    """
    if not type(b) is Blob:
        raise ValueError('`b` must be neoml.Blob.')

    if b.size == 0:
        raise ValueError("The blob mustn't be empty.")
    
    return Blob(PythonWrapper.blob_div(float(a), b._internal))

def max(a, b):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blob mustn't be empty.")
    
    return Blob(PythonWrapper.blob_max(a._internal, float(b)))

def max(a, b):
    """
    """
    if not type(b) is Blob:
        raise ValueError('`b` must be neoml.Blob.')

    if b.size == 0:
        raise ValueError("The blob mustn't be empty.")
    
    return Blob(PythonWrapper.blob_max(float(a), b._internal))

def sum(a):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_sum(a._internal))

def neg(a):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_neg(a._internal))

def abs(a):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_abs(a._internal))

def log(a):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_log(a._internal))

def exp(a):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if a.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_exp(a._internal))

def clip(blob, min_value, max_value):
    """
    """
    if not type(blob) is Blob:
        raise ValueError('`blob` must be neoml.Blob.')

    if blob.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_clip(blob._internal, float(min_value), float(max_value)))

def top_k(a, k=1):
    """
    """
    if not type(a) is Blob:
        raise ValueError('`a` must be neoml.Blob.')

    if int(k) <= 0:
        raise ValueError('`k` must be > 0.')

    if a.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_top_k(a._internal, int(k)))

def binary_cross_entropy(labels, preds, fromLogits):
    """
    """
    if not type(labels) is Blob:
        raise ValueError('`labels` must be neoml.Blob.')

    if not type(preds) is Blob:
        raise ValueError('`preds` must be neoml.Blob.')

    if labels.shape != preds.shape:
        raise ValueError("The blobs must have the same shape.")

    if labels.size == 0:
        raise ValueError("The blobs mustn't be empty.")
    
    return Blob(PythonWrapper.blob_binary_cross_entropy(labels._internal, preds._internal, bool(fromLogits)))
