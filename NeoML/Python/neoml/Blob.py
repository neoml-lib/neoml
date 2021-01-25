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
--------------------------------------------------------------------------------------------------------------
"""

import neoml.PythonWrapper as PythonWrapper
import numpy


class Blob:

    def __init__(self, internal):
        if not type(internal) is PythonWrapper.Blob:
            raise ValueError('The `blob` must be PythonWrapper.Blob.')
        self._internal = internal

    @property
    def shape(self):
        """
        """
        return self._internal.shape()

    @property
    def batch_len(self):
        """
        """
        return self._internal.batch_len()

    @property
    def batch_width(self):
        """
        """
        return self._internal.batch_width()

    @property
    def list_size(self):
        """
        """
        return self._internal.list_size()

    @property
    def height(self):
        """
        """
        return self._internal.height()

    @property
    def width(self):
        """
        """
        return self._internal.width()

    @property
    def depth(self):
        """
        """
        return self._internal.depth()

    @property
    def channels(self):
        """
        """
        return self._internal.channels()

    @property
    def size(self):
        """
        """
        return self._internal.size()

    @property
    def object_count(self):
        """
        """
        return self._internal.object_count()

    @property
    def object_size(self):
        """
        """
        return self._internal.object_size()

    @property
    def geometrical_size(self):
        """
        """
        return self._internal.geometrical_size()

    @property
    def data(self):
        """
        """
        return self._internal.data()

# -------------------------------------------------------------------------------------------------------------


def asblob(math_engine, data):
    """
    """
    np_data = numpy.asarray(data)

    if np_data.dtype != numpy.float32 and np_data.dtype != numpy.int32:
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')

    if len(np_data.shape) > 7:
        raise ValueError('The `data` must have < 8 dimensions.')

    return Blob(PythonWrapper.tensor(math_engine._internal, np_data))


def vector(math_engine, size, dtype):
    """
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if size < 1:
        raise ValueError('The `size` must be > 0.')

    shape = (size, 1, 1, 1, 1, 1, 1)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def matrix(math_engine, matrix_height, matrix_width, dtype="float32"):
    """
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if matrix_height < 1:
        raise ValueError('The `matrix_height` must be > 0.')
    if matrix_width < 1:
        raise ValueError('The `matrix_width` must be > 0.')

    shape = (matrix_height, matrix_width, 1, 1, 1, 1, 1)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def tensor(math_engine, shape, dtype="float32"):
    """
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if shape.size < 8:
        raise ValueError('The `shape.size` must be <= 7.')

    if numpy.any(shape <= 0):
        raise ValueError('All `shape` elements must be > 0.')

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def data_blob(math_engine, batch_len, batch_width, channels, dtype="float32", data=None):
    """
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if batch_len < 1:
        raise ValueError('The `batch_len` must be > 0.')
    if batch_width < 1:
        raise ValueError('The `batch_width` must be > 0.')
    if channels < 1:
        raise ValueError('The `channels` must be > 0.')

    shape = (channels, 1, 1, 1, 1, batch_width, batch_len)

    if data is None:
        return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))

    np_data = numpy.asarray( data )

    if np_data.size != batch_len * batch_width * channels :
        raise ValueError('The `data.size` must be equal to `batch_len * batch_width * channels`.')

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype, np_data))


def list_blob(math_engine, batch_len, batch_width, list_size, channels, dtype="float32"):
    """
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if batch_len < 1:
        raise ValueError('The `batch_len` must be > 0.')
    if batch_width < 1:
        raise ValueError('The `batch_width` must be > 0.')
    if list_size < 1:
        raise ValueError('The `list_size` must be > 0.')
    if channels < 1:
        raise ValueError('The `channels` must be > 0.')

    shape = (batch_len, batch_width, list_size, 1, 1, 1, channels)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def image2d(math_engine, batch_len, batch_width, height, width, channels, dtype="float32"):
    """
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if batch_len < 1:
        raise ValueError('The `batch_len` must be > 0.')
    if batch_width < 1:
        raise ValueError('The `batch_width` must be > 0.')
    if height < 1:
        raise ValueError('The `height` must be > 0.')
    if width < 1:
        raise ValueError('The `width` must be > 0.')
    if channels < 1:
        raise ValueError('The `channels` must be > 0.')

    shape = (batch_len, batch_width, 1, height, width, 1, channels)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def image3d(math_engine, batch_len, batch_width, height, width, depth, channels, dtype="float32"):
    """
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if batch_len < 1:
        raise ValueError('The `batch_len` must be > 0.')
    if batch_width < 1:
        raise ValueError('The `batch_width` must be > 0.')
    if height < 1:
        raise ValueError('The `height` must be > 0.')
    if width < 1:
        raise ValueError('The `width` must be > 0.')
    if depth < 1:
        raise ValueError('The `depth` must be > 0.')
    if channels < 1:
        raise ValueError('The `channels` must be > 0.')

    shape = (batch_len, batch_width, 1, height, width, depth, channels)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))
