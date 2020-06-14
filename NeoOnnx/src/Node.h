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

#include "NodeAttributes.h"
#include "Tensor.h"
#include "NeoOnnxCheck.h"

// Forward declaration(s).
namespace onnx {
class NodeProto;
class TensorProto;
class ValueInfoProto;
} // namespace onnx;

namespace NeoOnnx {

// Node in the ONNX calculation graph.
class CNode {
public:
	virtual ~CNode() = default;

	// Calculate output tensors shape and (if possible) data.
	virtual void OnnxReshape() = 0;

	// Marks ONNX tensors dimensions as blob dimensions from NeoML.
	virtual void MarkTensorDims() = 0;

	// Adds layers, representing this node, to the dnn (if such layers exist).
	virtual void AddLayers( CDnn& net ) = 0;

	// Gets the number of outputs.
	int OutputCount() const;

	// Information about input.
	struct CInputInfo {
		CInputInfo( CNode* inputNode, int outputIndex ) : InputNode( inputNode ), OutputIndex( outputIndex ) {}

		CNode* InputNode; // Node connected to this input.
		const int OutputIndex; // Node's output number connected to this input.
	};
	
	// Gets data of index'th input.
	const CTensor& InputTensor( int index ) const;
	CTensor& InputTensor( int index );

	// Fabric method. Creates CNode's derivative for given ONNX node.
	static CNode* CreateNode( const onnx::NodeProto& onnxNode, CMap<CString, CInputInfo>& nodeOutputs, IMathEngine& mathEngine );

protected:
	// Information about output.
	struct COutputInfo {
		// Constructor for ONNX node output, which doesn't with any NeoML layer's output.
		COutputInfo() : Layer( nullptr ), OutputIndex( NotFound ) {}

		// Constructor for ONNX node output, matching it with layer's outputIndex'th output.
		COutputInfo( CBaseLayer* layer, int outputIndex ) : Layer( layer ), OutputIndex( outputIndex )
			{ CheckNeoOnnxInternal( layer != nullptr, "non empty output info with layer == nullptr" ); }

		const CBaseLayer* Layer; // Used NeoML layer (nullptr if there is no layer mapped with this output)
		const int OutputIndex; // NeoML layer's output index, mapped with this output
	};

	const CNodeAttributes attributes; // Attributes of this node.
	const int onnxOutputCount;
	CArray<CTensor> outputData; // Node outputs.
	CArray<CInputInfo> input; // Node inputs.
	CArray<COutputInfo> outputInfo;
	const onnx::NodeProto& onnxNode; // Reference to ONNX node. Used for diagnostics.

	CNode( const onnx::NodeProto& node, CMap<CString, CInputInfo>& nodeOutputs );

	// Get info about output, connected to index'th input
	const COutputInfo& InputInfo( int index ) const;
	const CBaseLayer& InputLayer( int index ) const;
	int InputLayerIndex( int index ) const;
};

} // namespace NeoOnnx
