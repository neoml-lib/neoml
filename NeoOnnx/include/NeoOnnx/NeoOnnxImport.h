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
#include <NeoOnnx/TensorLayout.h>

namespace NeoOnnx {

// Additional settings for ONNX import
struct NEOONNX_API CImportSettings {
	// InputLayouts[Name][i] contains the blob dim which is used as i'th axis of input with Name
	// E.g. If ONNX has 4-dim input NCHW, and you want to feed NeoML blobs in NHWC format then add
	//    InputLayouts.Add( Name, { BD_BatchWidth, BD_Channels, BD_Height, BD_Width } );
	// If not set the default behavior is used (see LoadFromOnnx)
	CMap<CString, CTensorLayout> InputLayouts;

	// OutputLayouts[Name][i] contains the blob dim which is used as i'th axis of output with Name
	// E.g. If ONNX has 2-dim output and you want to put the first dim to batch and second to channels then add
	//    OutputLayouts.Add( Name, { BD_BatchWidth, BD_Channels } );
	// If not set the default behavior is used (see LoadFromOnnx)
	CMap<CString, CTensorLayout> OutputLayouts;

	// After the import the net is passed to NeoML::OptimizeDnn
	// The settings used by OptimizeDnn
	CDnnOptimizationSettings DnnOptimizationSettings{};
};

// Information about ONNX optimizations
struct NEOONNX_API COnnxOptimizationReport {
	int GELU = 0;
	int HardSigmoid = 0;
	int HSwish = 0;
	int LayerNorm = 0;
	int SqueezeAndExcite = 0;
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
	// During export 2 types of optimization are performed
	// 1. ONNX-only optimizations. Mostly they just removed artifacts from ONNX generation. See OnnxOptimizationReport.
	// 2. NeoML::OptimizeDnn. See OptimizationReport.
	COnnxOptimizationReport OnnxOptimizationReport;
	CDnnOptimizationReport OptimizationReport;
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
// This rule may be overridden by using CImportSettings::InputLayouts and CImportSettings::OutputLayouts
//
// Throw std::logic_error if failed to load network


// Loads network "dnn" from onnx file "fileName"
NEOONNX_API void LoadFromOnnx( const char* fileName, const CImportSettings& settings,
	NeoML::CDnn& dnn, CImportedModelInfo& info );

// Loads network "dnn" from buffer with onnx data
NEOONNX_API void LoadFromOnnx( const void* buffer, int bufferSize, const CImportSettings& settings,
	NeoML::CDnn& dnn, CImportedModelInfo& info );

} // namespace NeoOnnx

