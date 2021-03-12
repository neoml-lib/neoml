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
import neoml.MathEngine as MathEngine
import numpy


class Blob:

    def __init__(self, internal):
        """
        """
        if not type(internal) is PythonWrapper.Blob:
            raise ValueError('The `blob` must be PythonWrapper.Blob.')
        self._internal = internal

    @property
    def math_engine(self):
        """
        """
        return self._internal.math_engine()

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

    def asarray(self, copy=False):
        """
        """
        if type(self.math_engine) is MathEngine.CpuMathEngine:
            return numpy.array(self._internal, copy=copy)
        cpu_blob = self.copy(MathEngine.default_math_engine())
        return numpy.array(cpu_blob._internal, copy=copy)

    def copy(self, math_engine):
        """
        """
        return Blob(self._internal.copy(math_engine._internal))

# -------------------------------------------------------------------------------------------------------------

def store(blob, file_path):
    """
    """
    if not type(blob) is Blob:
        raise ValueError('The `blob` must be neoml.Blob.')

    if blob.size == 0:
        raise ValueError("The `blob` can't be empty.")

    PythonWrapper.store_blob(blob._internal, str(file_path))


def load(math_engine, file_path):
    """
    """
    if not isinstance(math_engine, MathEngine.MathEngine):
        raise ValueError('The `math_engine` must be neoml.MathEngine.')

    return Blob(PythonWrapper.load_blob(math_engine._internal, str(file_path)))


def asblob(math_engine, data, shape=None, copy=False):
    """
    """
    if shape is None:
        shape = numpy.ones(7, numpy.int32)
    else:
        shape = numpy.array(shape, dtype=numpy.int32, copy=False)

    if len(shape) != 7:
        raise ValueError('The `shape` must have 7 dimension sizes.')

    np_data = numpy.array(data, copy=False, order='C')

    if len(np_data.shape) > 7:
        raise ValueError('The `shape` must have not more then 7 dimensions.')

    dtype = 'none'
    if np_data.dtype == numpy.float32:
        dtype = 'float32'
    elif np_data.dtype == numpy.int32:
        dtype = 'int32'
    else:
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')

    if type(math_engine) is MathEngine.GpuMathEngine:
        copy = True

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype, np_data, bool(copy)))


def vector(math_engine, size, dtype="float32"):
    """
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if size < 1:
        raise ValueError('The `size` must be > 0.')

    shape = numpy.array((size, 1, 1, 1, 1, 1, 1), dtype=numpy.int32)

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

    shape = numpy.array((matrix_height, matrix_width, 1, 1, 1, 1, 1), dtype=numpy.int32)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def tensor(math_engine, shape, dtype="float32"):
    """
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')

    shape = numpy.array(shape, dtype=numpy.int32, copy=False)

    if shape.size != 7:
        raise ValueError('The `shape.size` must be == 7.')

    if numpy.any(shape <= 0):
        raise ValueError('All `shape` elements must be > 0.')

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


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

    shape = numpy.array((batch_len, batch_width, list_size, 1, 1, 1, channels), dtype=numpy.int32, copy=False)

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

    shape = numpy.array((batch_len, batch_width, 1, height, width, 1, channels), dtype=numpy.int32, copy=False)

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

    shape = numpy.array((batch_len, batch_width, 1, height, width, depth, channels), dtype=numpy.int32, copy=False)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))
