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
    """The class that stores and transmits data in neural networks.
    
    A blob is a 7-dimensional array, with each of its dimensions assigned
    a specific meaning:
    - BatchLength is a "time" axis, used to denote data sequences; 
        it is mainly used in recurrent networks
    - BatchWidth corresponds to the batch, used to pass several independent
        objects together
    - ListSize is the dimensions for the objects that are connected
        (for example, pixels out of one image) but do not form a sequence
    - Height is the height of a matrix or an image
    - Width is the width of a matrix or an image
    - Depth is the width of a 3-dimensional image
    - Channels corresponds to channels for multi-channel image formats 
        and is also used to work with one-dimensional vectors
    
    Data types
    ---------
    int and float data types are supported. Both are 32-bit.
    """

    def __init__(self, internal):
        """Creates a new blob.
        """
        if not type(internal) is PythonWrapper.Blob:
            raise ValueError('The `blob` must be PythonWrapper.Blob.')
        self._internal = internal

    @property
    def math_engine(self):
        """The math engine that allocates the blob memory 
        and will work with its data.
        """
        return self._internal.math_engine()

    @property
    def shape(self):
        """The blob shape: the array of 7 int elements with dimension lengths.
        """
        return self._internal.shape()

    @property
    def batch_len(self):
        """Returns the BatchLength dimension.
        """
        return self._internal.batch_len()

    @property
    def batch_width(self):
        """Returns the BatchWidth dimension.
        """
        return self._internal.batch_width()

    @property
    def list_size(self):
        """Returns the ListSize dimension.
        """
        return self._internal.list_size()

    @property
    def height(self):
        """Returns the Height dimension.
        """
        return self._internal.height()

    @property
    def width(self):
        """Returns the Width dimension.
        """
        return self._internal.width()

    @property
    def depth(self):
        """Returns the Depth dimension.
        """
        return self._internal.depth()

    @property
    def channels(self):
        """Returns the Channels dimension.
        """
        return self._internal.channels()

    @property
    def size(self):
        """Returns the blob total size: the product of all dimensions.
        """
        return self._internal.size()

    @property
    def object_count(self):
        """Returns the number of objects in the blob, 
        equal to BatchLength * BatchWidth * ListSize.
        """
        return self._internal.object_count()

    @property
    def object_size(self):
        """Returns the size of one object, equal to
        Height * Width * Depth * Channels.
        """
        return self._internal.object_size()

    def asarray(self, copy=False):
        """Returns the contents of the blob as a multi-dimensional array, 
        keeping only those dimensions that are more than 1 element long.
        If all dimensions are 1, the blob will be a one-element array.
        
        Parameters
        ---------
        copy : bool, default=False
            If True, the data will be copied. If False, the array may share
            the memory buffer with the blob if possible and only provide 
            more convenient access to the same data.
            Not copying may be impossible if the blob is in GPU memory.
        """
        if type(self.math_engine) is MathEngine.CpuMathEngine:
            return numpy.array(self._internal, copy=copy)
        cpu_blob = self.copy(MathEngine.default_math_engine())
        return numpy.array(cpu_blob._internal, copy=False)

    def copy(self, math_engine):
        """Creates a blob copy independent of this blob.
        """
        return Blob(self._internal.copy(math_engine._internal))

# -------------------------------------------------------------------------------------------------------------

def store(blob, file_path):
    """Stores the blob in a file at the specified path.
    
    Parameters
    ---------
    file_path : str
        The full path to the file where the blob should be stored.
    """
    if not type(blob) is Blob:
        raise ValueError('The `blob` must be neoml.Blob.')

    if blob.size == 0:
        raise ValueError("The `blob` can't be empty.")

    PythonWrapper.store_blob(blob._internal, str(file_path))


def load(math_engine, file_path):
    """Loads the blob from the specified location.
    
    Parameters
    ---------
    math_engine : object
        The math engine that works with this blob.
    file_path : str
        The full path to the file from which the blob should be loaded.
    """
    if not isinstance(math_engine, MathEngine.MathEngine):
        raise ValueError('The `math_engine` must be neoml.MathEngine.')

    return Blob(PythonWrapper.load_blob(math_engine._internal, str(file_path)))


def asblob(math_engine, data, shape=None, copy=False):
    """Organizes the data from a memory buffer into a blob.
    
    Parameters
    ---------
    math_engine : object
        The math engine that works with this blob.
    data : object
        A pointer to the data.
    shape : array of int, default=None
        The target blob dimensions. If none are specified,
        a one-element blob will be assumed.
    copy : bool, default=False
        Specifies if the data should be copied to another memory block
        or kept in the same place if possible, making the data parameter 
        point to the body of the newly-created blob.
        Not copying may be impossible if the blob is in GPU memory.
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
    """Creates a one-dimensional blob, that is, a vector.
    
    Parameters
    ---------
    math_engine : object
        The math engine that works with this blob.
    size : int, > 0
        The vector length.
    dtype : {"float32", "int32"}, default="float32"
        The type of data in the blob.
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if size < 1:
        raise ValueError('The `size` must be > 0.')

    shape = numpy.array((size, 1, 1, 1, 1, 1, 1), dtype=numpy.int32)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def matrix(math_engine, matrix_height, matrix_width, dtype="float32"):
    """Creates a two-dimensional blob, that is, a matrix.
    
    Parameters
    ---------
    math_engine : object
        The math engine that works with this blob.
    matrix_height : int, > 0
        The matrix height.
    matrix_width : int, > 0
        The matrix width.
    dtype : {"float32", "int32"}, default="float32"
        The type of data in the blob.
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
    """Creates a blob of the specified shape.
    
    Parameters
    ---------
    math_engine : object
        The math engine that works with this blob.
    shape : array of int
        The target blob dimensions.
    dtype : {"float32", "int32"}, default="float32"
        The type of data in the blob.
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
    """Creates a blob with one-dimensional Height * Width * Depth elements.
    
    Parameters
    ---------
    math_engine : object
        The math engine that works with this blob.
    batch_len : int, > 0
        The BatchLength dimension of the new blob.
    batch_width : int, > 0
        The BatchWidth dimension of the new blob.
    list_size : int, > 0
        The ListSize dimension of the new blob.
    channels : int, > 0
        The Channels dimension of the new blob.
    dtype : {"float32", "int32"}, default="float32"
        The type of data in the blob.
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
    """Creates a blob with two-dimensional multi-channel images.
    
    Parameters
    ---------
    math_engine : object
        The math engine that works with this blob.
    batch_len : int, > 0
        The BatchLength dimension of the new blob.
    batch_width : int, > 0
        The BatchWidth dimension of the new blob.
    height : int, > 0
        The image height.
    width: int, > 0
        The image width.
    channels : int, > 0
        The number of channels in the image format.
    dtype : {"float32", "int32"}, default="float32"
        The type of data in the blob.
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
    """Creates a blob with three-dimensional multi-channel images.
    
    Parameters
    ---------
    math_engine : object
        The math engine that works with this blob.
    batch_len : int, > 0
        The BatchLength dimension of the new blob.
    batch_width : int, > 0
        The BatchWidth dimension of the new blob.
    height : int, > 0
        The image height.
    width: int, > 0
        The image width.
    depth: int, > 0
        The image depth.
    channels : int, > 0
        The number of channels in the image format.
    dtype : {"float32", "int32"}, default="float32"
        The type of data in the blob.
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
