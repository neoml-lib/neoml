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

import neoml.PythonWrapper as PythonWrapper
import neoml.Dnn


class ImportedModelInputInfo:
	__slots__ = ['name']
	def __init__(self, name):
		self.name = name


class ImportedModelOutputInfo:
	__slots__ = ['name', 'dim_count']
	def __init__(self, name, dim_count):
		self.name = name
		self.dim_count = dim_count


class ImportedModelInfo:
	__slots__ = ['inputs', 'outputs', 'metadata']
	def __init__(self):
		self.inputs = list()
		self.outputs = list()
		self.metadata = dict()


def load_from_file(file_name, math_engine):
	dnn = neoml.Dnn.Dnn(math_engine)
	model_info = PythonWrapper.load_onnx_from_file(file_name, dnn)
	return dnn, model_info


def load_from_buffer(buffer, math_engine):
	dnn = neoml.Dnn.Dnn(math_engine)
	model_info = PythonWrapper.load_onnx_from_buffer(buffer, dnn)
	return dnn, model_info