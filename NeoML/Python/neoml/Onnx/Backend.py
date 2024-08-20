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

from .API import load_from_buffer
import onnx.backend.base
import neoml.MathEngine
import neoml.Blob
import numpy as np


_onnx_type_to_np = [
    None,  # onnx::TensorProto::UNDEFINED
    np.float32,  # FLOAT
    np.uint8,  # UINT8
    np.int8,  # INT8
    np.uint16,  # UINT16
    np.int16,  # INT16
    np.int32,  # INT32
    np.int64,  # INT64
    None,  # STRING
    bool,  # BOOL
    np.float16,  # FLOAT16
    np.double,  # DOUBLE
    np.uint32,  # UINT32
    np.uint64,  # UINT64
    None,  # COMPLEX64
    None,  # COMPLEX64
    None  # BFLOAT16
]


class BackendRep:
    def __init__(self, model, device):
        if device == 'CPU':
            math_engine = neoml.MathEngine.CpuMathEngine(1)
        else:
            math_engine = neoml.MathEngine.GpuMathEngine()
        self.dnn, self.info = load_from_buffer(model.SerializeToString(), math_engine)
        output_infos = self.info['outputs']
        self.output_dtypes = [None] * len(output_infos)
        self.output_dim_count = [None] * len(output_infos)
        for value_info in model.graph.output:
            for idx, out in enumerate(output_infos):
                if out == value_info.name:
                    try:
                        self.output_dtypes[idx] = \
                            _onnx_type_to_np[value_info.type.tensor_type.elem_type]
                    except:
                        pass
                    try:
                        self.output_dim_count[idx] = \
                            len(value_info.type.tensor_type.shape.dim)
                    except:
                        pass

    def run(self, inputs, **kwargs):
        neoml_inputs = dict()
        def _get_dtype(orig_dtype):
            if np.issubdtype(orig_dtype, np.floating):
                return np.float32
            elif np.issubdtype(orig_dtype, np.integer) or orig_dtype == bool:
                return np.int32
            raise ValueError(f'{orig_dtype} is not supported by neoml.Onnx.BackendRep')
        input_infos = self.info['inputs']
        for idx, onnx_input in enumerate(inputs):
            onnx_input = onnx_input.astype(_get_dtype(onnx_input.dtype), copy=False)
            neoml_shape = list(onnx_input.shape) + [1] * (7 - len(onnx_input.shape))
            neoml_blob = neoml.Blob.asblob(self.dnn.math_engine, onnx_input, neoml_shape)
            neoml_inputs[input_infos[idx]] = neoml_blob
        neoml_outputs = self.dnn.run(neoml_inputs)
        result = list()
        for output, onnx_dtype, dim_count in zip(self.info['outputs'], self.output_dtypes, self.output_dim_count):
            out_blob = neoml_outputs[output]
            result.append(out_blob.asarray())
            if dim_count is not None:
                result[-1].resize(out_blob.shape[:dim_count])
            if onnx_dtype is not None:
                result[-1] = result[-1].astype(onnx_dtype, copy=False)
        return result


class Backend(onnx.backend.base.Backend):
    @classmethod
    def prepare(cls, model, device='CPU'):
        return BackendRep(model, device)
    
    @classmethod
    def run_model(cls, model, inputs, device='CPU', **kwargs):
        back_rep = cls.prepare(model, device)
        return back_rep.run(inputs)

    @classmethod
    def supports_device(cls, device):
        device = device.upper()
        return device == 'CPU' or device == 'GPU' or device == 'CUDA'


run_model = Backend.run_model
supports_device = Backend.supports_device
prepare = Backend.prepare
