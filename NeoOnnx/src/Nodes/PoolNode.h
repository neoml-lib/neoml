/* Copyright Â© 2017-2020 ABBYY Production LLC

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

#include "../Node.h"
#include "../NodeUtils.h"

namespace NeoOnnx {

// Base class for non-global Pool operator nodes
class CPoolNodeBase : public COpNode {
public:
	CPoolNodeBase( TPoolingType poolingType, int nodeIndex, const onnx::NodeProto& pool, int opsetVersion );

	// CNode methods' realizations
	void CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine ) override;
	void LabelTensorDims( const CTensorCache& tensors, CDimCache& dims ) override;
	void AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
		CNeoMLLinkCache& neoMLLinks, CDnn& dnn ) override;

private:
	TPoolingType poolingType; // pooling type
	const CString autoPad; // padding mode
	CFastArray<int, 8> kernelShape; // shape of pool kernel
	CFastArray<int, 8> strides; // kernel strides
	CFastArray<int, 8> pads; // convolution paddings
};

// MaxPool operator node
class CMaxPoolNode : public CPoolNodeBase {
public:
	CMaxPoolNode( int nodeIndex, const onnx::NodeProto& maxPool, int opsetVersion ) :
		CPoolNodeBase( PT_Max, nodeIndex, maxPool, opsetVersion ) {}
};

// AveragePool operator node
class CAveragePoolNode : public CPoolNodeBase {
public:
	CAveragePoolNode( int nodeIndex, const onnx::NodeProto& averagePool, int opsetVersion ) :
		CPoolNodeBase( PT_Mean, nodeIndex, averagePool, opsetVersion ) {}
};

} // namespace NeoOnnx