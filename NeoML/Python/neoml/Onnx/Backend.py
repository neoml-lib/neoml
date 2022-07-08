""" Copyright (c) 2017-2022 ABBYY Production LLC

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

class BackendRep:
	def __init__(self, model, device):
		if device == 'CPU':
			math_engine = neoml.MathEngine.CpuMathEngine(1)
		else:
			math_engine = neoml.MathEngine.GpuMathEngine()
		self.dnn, self.info = load_from_buffer(model.SerializeToString(), math_engine)
	
	def run(self, inputs, **kwargs):
		neoml_inputs = dict()
		for idx, onnx_input in enumerate(inputs):
			neoml_shape = list(onnx_input.shape) + [1] * (7 - len(onnx_input.shape))
			neoml_blob = neoml.Blob.asblob(self.dnn.math_engine, onnx_input, neoml_shape)
			neoml_inputs[self.info.inputs[idx].name] = neoml_blob
		neoml_outputs = self.dnn.run(neoml_inputs)
		result = list()
		for output in self.info.outputs:
			out_blob = neoml_outputs[output.name]
			result.append(out_blob.asarray())
			result[-1].resize(out_blob.shape[:output.dim_count])
		return result


class Backend(onnx.backend.base.Backend):
	@classmethod
	def prepare(cls, model, device='CPU'):
		return BackendRep(model, device)
	
	@classmethod
	def run_model(cls, model, inputs, device='CPU', **kwargs):
		back_rep = cls.prepare(model, device)
		assert back_rep is not NesterovGradient
		return back_rep.run(inputs)

	@classmethod
	def supports_device(cls, device):
		device = device.upper()
		return device == 'CPU' or device == 'GPU' or device == 'CUDA'


run_model = Backend.run_model
supports_device = Backend.supports_device
prepare = Backend.prepare
