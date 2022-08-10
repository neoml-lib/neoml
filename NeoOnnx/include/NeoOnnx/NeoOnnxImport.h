/* Copyright © 2017-2020 ABBYY Production LLC

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

#pragma once

#include <NeoOnnx/NeoOnnxDefs.h>
#include <NeoML/NeoML.h>

namespace NeoOnnx {

// Additional settings for ONNX import
struct NEOONNX_API CImportSettings {
};

// Information about imported model
struct NEOONNX_API CImportedModelInfo {
	struct NEOONNX_API CInputInfo {
		CString Name;
	};

	struct NEOONNX_API COutputInfo {
		CString Name;
	};

	CArray<CInputInfo> Inputs;
	CArray<COutputInfo> Outputs;
	CMap<CString, CString> Metadata; // metadata_props
};

// The load functions build CDnn based on ONNX in the following way:
//
// For every uninitialized onnx graph input there will be CSourceLayer with the same name
// For every CSourceLayer will be allocated input blob of the given size
// Inputs with initializers will be ignored (used for parameters calculation)
//
// For every onnx graph output there will be CSinkLayer with the same name
//
// info will be filled with information about model's inputs, outputs, metadata etc.
//
// Input and output blobs have the following relations with the ONNX N-dimensional tensors:
// - first N dimensions of the blob are corresponding to the N dimensions of the ONNX tensor
// - other dimensions must be of length 1
// E.g. 4-dimensional ONNX tensor [1, 3, 224, 224] is equivalent to a NeoML blob of shape [1, 3, 224, 224, 1, 1, 1]
// In C++ use CDnnBlob::CreateTensor( ... , onnxShape ); function
// In Python use neomlBlobShape = onnxShape + [1] * (7 - len(onnxShape))
//
// Throw std::logic_error if failed to load network


// Loads network "dnn" from onnx file "fileName"
NEOONNX_API void LoadFromOnnx( const char* fileName, const CImportSettings& settings,
	NeoML::CDnn& dnn, CImportedModelInfo& info );

// Loads network "dnn" from buffer with onnx data
NEOONNX_API void LoadFromOnnx( const void* buffer, int bufferSize, const CImportSettings& settings,
	NeoML::CDnn& dnn, CImportedModelInfo& info );

} // namespace NeoOnnx

