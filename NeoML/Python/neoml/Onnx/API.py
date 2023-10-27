""" Copyright (c) 2017-2023 ABBYY

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
import neoml.Dnn


def _validate_layouts(layouts):
    if layouts is None:
        return  # Everything is OK, default behavior
    for name, layout in layouts.items():
        if not isinstance(name, str):
            raise ValueError('name must be a str')
        rem_dims = {'batch_length', 'batch_width', 'list_size', 'height', 'width', 'depth', 'channels'}
        for dim in layout:
            if not dim in rem_dims:
                raise ValueError('illegal or double dim in layout:' + str(dim))
            rem_dims.remove(dim)


def load_from_file(file_name, math_engine, input_layouts=None, output_layouts=None):
    _validate_layouts(input_layouts)
    _validate_layouts(output_layouts)
    dnn = neoml.Dnn.Dnn(math_engine)
    model_info = PythonWrapper.load_onnx_from_file(file_name, dnn, input_layouts, output_layouts)
    return dnn, model_info


def load_from_buffer(buffer, math_engine, input_layouts=None, output_layouts=None):
    _validate_layouts(input_layouts)
    _validate_layouts(output_layouts)
    dnn = neoml.Dnn.Dnn(math_engine)
    model_info = PythonWrapper.load_onnx_from_buffer(buffer, dnn, input_layouts, output_layouts)
    return dnn, model_info
