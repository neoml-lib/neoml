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

import neoml
import neoml.PythonWrapper as PythonWrapper
import neoml.MathEngine as MathEngine
import numpy as np


class Blob:
    """The class that stores and transmits data in neural networks.
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
        """The **BatchLength** dimension.
        """
        return self._internal.batch_len()

    @property
    def batch_width(self):
        """The **BatchWidth** dimension.
        """
        return self._internal.batch_width()

    @property
    def list_size(self):
        """The **ListSize** dimension.
        """
        return self._internal.list_size()

    @property
    def height(self):
        """The **Height** dimension.
        """
        return self._internal.height()

    @property
    def width(self):
        """The **Width** dimension.
        """
        return self._internal.width()

    @property
    def depth(self):
        """The **Depth** dimension.
        """
        return self._internal.depth()

    @property
    def channels(self):
        """The **Channels** dimension.
        """
        return self._internal.channels()

    @property
    def size(self):
        """The blob total size: the product of all dimensions.
        """
        return self._internal.size()

    @property
    def object_count(self):
        """The number of objects in the blob, 
        equal to **BatchLength** * **BatchWidth** * **ListSize**.
        """
        return self._internal.object_count()

    @property
    def object_size(self):
        """The size of one object, equal to
        **Height** * **Width** * **Depth** * **Channels**.
        """
        return self._internal.object_size()

    def asarray(self, copy=False, keep_dims=False):
        """Returns the contents of the blob as a multi-dimensional array.

        :param copy: if `True`, the data will be copied. If `False`, the array may share
            the memory buffer with the blob if possible and only provide 
            more convenient access to the same data.
            Not copying may be impossible if the blob is in GPU memory.
        :type copy: bool, default=False

        :param keep_dims: if `True` then the result shape will contain all the 7
        dimensions of the blob. Otherwise it will contain only dimensions longer than 1.
        If `False` and blob consists only of one element, then the result will be of shape (1,).
        :type keep_dims: bool, default=False
        """
        if type(self.math_engine) is MathEngine.CpuMathEngine:
            result = np.array(self._internal, copy=copy)
        else:
            cpu_blob = self.copy(MathEngine.default_math_engine())
            result = np.array(cpu_blob._internal, copy=False)
        if keep_dims:
            result.resize(self.shape)
        return result

    def copy(self, math_engine):
        """Creates a blob copy independent of this blob.
        """
        return Blob(self._internal.copy(math_engine._internal))

    def __add__(self, other):
        """Elementwise adds two blobs or a blob and a scalar value.
        """
        if self.size == 0:
            raise ValueError("The blob shouldn't be empty.")

        if type(other) is Blob:
            if not neoml.Utils.check_can_broadcast(self, other):
                raise ValueError("The blobs have incompatible shapes.")
            return Blob(PythonWrapper.blob_add(self._internal, other._internal))

        return Blob(PythonWrapper.blob_add(self._internal, float(other)))

    def __radd__(self, other):
        """Elementwise adds two blobs or a scalar value and a blob.
        """
        if self.size == 0:
            raise ValueError("The blob shouldn't be empty.")
        return Blob(PythonWrapper.blob_add(self._internal, float(other)))

    def __sub__(self, other):
        """Elementwise subtracts two blobs or a blob and a scalar value.
        """
        if self.size == 0:
            raise ValueError("The blob shouldn't be empty.")

        if type(other) is Blob:
            if not neoml.Utils.check_can_broadcast(self, other):
                raise ValueError("The blobs have incompatible shapes.")
            return Blob(PythonWrapper.blob_sub(self._internal, other._internal))

        return Blob(PythonWrapper.blob_sub(self._internal, float(other)))

    def __rsub__(self, other):
        """Elementwise subtracts two blobs or a scalar value and a blob.
        """
        if self.size == 0:
            raise ValueError("The blob mustn't be empty.")
        return Blob(PythonWrapper.blob_sub(float(other), self._internal))

    def __mul__(self, other):
        """Elementwise multiplies two blobs or a blob and a scalar value.
        """
        if self.size == 0:
            raise ValueError("The blob mustn't be empty.")

        if type(other) is Blob:
            if not neoml.Utils.check_can_broadcast(self, other):
                raise ValueError("The blobs have incompatible shapes.")
            return Blob(PythonWrapper.blob_mul(self._internal, other._internal))

        return Blob(PythonWrapper.blob_mul(self._internal, float(other)))

    def __rmul__(self, other):
        """Elementwise multiplies two blobs or a scalar value and a blob.
        """
        if self.size == 0:
            raise ValueError("The blob mustn't be empty.")
        return Blob(PythonWrapper.blob_mul(self._internal, float(other)))

    def __truediv__(self, other):
        """Elementwise divides two blobs or a blob and a scalar value.
        """
        if self.size == 0:
            raise ValueError("The blob mustn't be empty.")

        if type(other) is Blob:
            if not neoml.Utils.check_can_broadcast(self, other):
                raise ValueError("The blobs have incompatible shapes.")
            return Blob(PythonWrapper.blob_div(self._internal, other._internal))

        return Blob(PythonWrapper.blob_div(self._internal, float(other)))

    def __rtruediv__(self, other):
        """Elementwise divides two blobs or a scalar value and a blob.
        """
        if self.size == 0:
            raise ValueError("The blob mustn't be empty.")
        return Blob(PythonWrapper.blob_div(float(other), self._internal))

    def __neg__(self):
        """Takes elementwise negative of the blob.
        """
        if self.size == 0:
            raise ValueError("The blobs mustn't be empty.")
        return Blob(PythonWrapper.blob_neg(self._internal))

    def __lt__(self, other):
        """Returns self < other elementwise.
        """
        if self.size == 0:
            raise ValueError("The blob mustn't be empty.")

        if type(other) is Blob:
            if self.shape != other.shape:
                raise ValueError("The blobs have incompatible shapes.")
            return Blob(PythonWrapper.blob_less(self._internal, other._internal))

        return Blob(PythonWrapper.blob_less(self._internal, float(other)))

    def __gt__(self, other):
        """Returns self > other elementwise.
        """
        if self.size == 0:
            raise ValueError("The blob mustn't be empty.")

        if type(other) is Blob:
            if self.shape != other.shape:
                raise ValueError("The blobs have incompatible shapes.")
            return Blob(PythonWrapper.blob_less(other._internal, self._internal))

        return Blob(PythonWrapper.blob_less(float(other), self._internal))

    def __pow__(self, other):
        """Computes the power of self to other.
        """
        if self.size == 0:
            raise ValueError("The blob mustn't be empty.")

        if type(other) is Blob:
            if not neoml.Utils.check_can_broadcast(self, other):
                raise ValueError("The blobs have incompatible shapes.")
            return Blob(PythonWrapper.blob_pow(self._internal, other._internal))

        return Blob(PythonWrapper.blob_pow(self._internal, float(other)))

    def __rpow__(self, other):
        """Computes the power of self to other.
        """
        if self.size == 0:
            raise ValueError("The blob mustn't be empty.")
        return Blob(PythonWrapper.blob_pow(float(other), self._internal))

# -------------------------------------------------------------------------------------------------------------

def store(blob, file_path):
    """Stores the blob in a file at the specified path.

    :param file_path: the full path to the file where the blob should be stored.
    :type file_path: str
    """
    if not type(blob) is Blob:
        raise ValueError('The `blob` must be neoml.Blob.')

    if blob.size == 0:
        raise ValueError("The `blob` can't be empty.")

    PythonWrapper.store_blob(blob._internal, str(file_path))


def load(math_engine, file_path):
    """Loads the blob from the specified location.

    :param neoml.MathEngine.MathEngine math_engine: the math engine that works with this blob.

    :param file_path: the full path to the file from which the blob should be loaded.
    :type file_path: str
    """
    if not isinstance(math_engine, MathEngine.MathEngine):
        raise ValueError('The `math_engine` must be neoml.MathEngine.')

    return Blob(PythonWrapper.load_blob(math_engine._internal, str(file_path)))


def asblob(math_engine, data, shape=None, copy=False):
    """Organizes the data from a memory buffer into a blob.

    :param neoml.MathEngine.MathEngine math_engine: the math engine that works with this blob.

    :param data: a pointer to the data.
    :type data: object

    :param shape: the target blob dimensions. If none are specified,
        a one-element blob will be assumed.
    :type shape: array of int, default=None

    :param copy: specifies if the data should be copied to another memory block
        or kept in the same place if possible, making the data parameter 
        point to the body of the newly-created blob.
        Not copying may be impossible if the blob is in GPU memory.
    :type copy: bool, default=False
    """
    if shape is None:
        shape = np.ones(7, np.int32)
    else:
        shape = np.array(shape, dtype=np.int32, copy=False)

    if len(shape) != 7:
        raise ValueError('The `shape` must have 7 dimension sizes.')

    np_data = np.array(data, copy=False, order='C')

    if len(np_data.shape) > 7:
        raise ValueError('The `shape` must have not more then 7 dimensions.')

    if np.prod(np_data.shape) != np.prod(shape):
        raise ValueError('The blob must have as many elements as ndarray')

    dtype = 'none'
    if np_data.dtype == np.float32:
        dtype = 'float32'
    elif np_data.dtype == np.int32:
        dtype = 'int32'
    else:
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')

    if type(math_engine) is MathEngine.GpuMathEngine:
        copy = True

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype, np_data, bool(copy)))


def vector(math_engine, size, dtype="float32"):
    """Creates a one-dimensional blob, that is, a vector.

    :param neoml.MathEngine.MathEngine math_engine: the math engine that works with this blob.

    :param size: the vector length.
    :type size: int, > 0

    :param dtype: the type of data in the blob.
    :type dtype: str, {"float32", "int32"}, default="float32"
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if size < 1:
        raise ValueError('The `size` must be > 0.')

    shape = np.array((size, 1, 1, 1, 1, 1, 1), dtype=np.int32)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def matrix(math_engine, matrix_height, matrix_width, dtype="float32"):
    """Creates a two-dimensional blob, that is, a matrix.

    :param neoml.MathEngine.MathEngine math_engine: the math engine that works with this blob.

    :param matrix_height: the matrix height.
    :type matrix_height: int, > 0

    :param matrix_width: the matrix width.
    :type matrix_width: int, > 0

    :param dtype: the type of data in the blob.
    :type dtype: str, {"float32", "int32"}, default="float32"
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')
    if matrix_height < 1:
        raise ValueError('The `matrix_height` must be > 0.')
    if matrix_width < 1:
        raise ValueError('The `matrix_width` must be > 0.')

    shape = np.array((matrix_height, matrix_width, 1, 1, 1, 1, 1), dtype=np.int32)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def tensor(math_engine, shape, dtype="float32"):
    """Creates a blob of the specified shape.

    :param neoml.MathEngine.MathEngine math_engine: the math engine that works with this blob.

    :param shape: the target blob dimensions.
    :type shape: array of int

    :param dtype: the type of data in the blob.
    :type dtype: str, {"float32", "int32"}, default="float32"
    """
    if dtype != "float32" and dtype != "int32":
        raise ValueError('The `dtype` must be one of {`float32`, `int32`}.')

    shape = np.array(shape, dtype=np.int32, copy=False)

    if shape.size != 7:
        raise ValueError('The `shape.size` must be == 7.')

    if np.any(shape <= 0):
        raise ValueError('All `shape` elements must be > 0.')

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def list_blob(math_engine, batch_len, batch_width, list_size, channels, dtype="float32"):
    """Creates a blob with one-dimensional **Height** * **Width** * **Depth** elements.

    :param neoml.MathEngine.MathEngine math_engine: the math engine that works with this blob.

    :param batch_len: the **BatchLength** dimension of the new blob.
    :type batch_len: int, > 0

    :param batch_width: the **BatchWidth** dimension of the new blob.
    :type batch_width: int, > 0

    :param list_size: the **ListSize** dimension of the new blob.
    :type list_size: int, > 0

    :param channels: the **Channels** dimension of the new blob.
    :type channels: int, > 0

    :param dtype: the type of data in the blob.
    :type dtype: str, {"float32", "int32"}, default="float32"
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

    shape = np.array((batch_len, batch_width, list_size, 1, 1, 1, channels), dtype=np.int32, copy=False)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def image2d(math_engine, batch_len, batch_width, height, width, channels, dtype="float32"):
    """Creates a blob with two-dimensional multi-channel images.

    :param neoml.MathEngine.MathEngine math_engine: the math engine that works with this blob.

    :param batch_len: the **BatchLength** dimension of the new blob.
    :type batch_len: int, > 0

    :param batch_width: the **BatchWidth** dimension of the new blob.
    :type batch_width: int, > 0

    :param height: the image height.
    :type height: int, > 0

    :param width: the image width.
    :type width: int, > 0

    :param channels: the number of channels in the image format.
    :type channels: int, > 0

    :param dtype: the type of data in the blob.
    :type dtype: str, {"float32", "int32"}, default="float32"
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

    shape = np.array((batch_len, batch_width, 1, height, width, 1, channels), dtype=np.int32, copy=False)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))


def image3d(math_engine, batch_len, batch_width, height, width, depth, channels, dtype="float32"):
    """Creates a blob with three-dimensional multi-channel images.

    :param neoml.MathEngine.MathEngine math_engine: the math engine that works with this blob.

    :param batch_len: the **BatchLength** dimension of the new blob.
    :type batch_len: int, > 0

    :param batch_width: the **BatchWidth** dimension of the new blob.
    :type batch_width: int, > 0

    :param height: the image height.
    :type height: int, > 0

    :param width: the image width.
    :type width: int, > 0

    :param depth: the image depth.
    :type depth: int, > 0

    :param channels: the number of channels in the image format.
    :type channels: int, > 0

    :param dtype: the type of data in the blob.
    :type dtype: str, {"float32", "int32"}, default="float32"
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

    shape = np.array((batch_len, batch_width, 1, height, width, depth, channels), dtype=np.int32, copy=False)

    return Blob(PythonWrapper.tensor(math_engine._internal, shape, dtype))
