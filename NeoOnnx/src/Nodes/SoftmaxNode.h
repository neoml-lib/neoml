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

#include "../Node.h"

namespace NeoOnnx {

// Softmax operator graph node
class CSoftmaxNode : public COpNode {
public:
	CSoftmaxNode( int nodeIndex, const onnx::NodeProto& softmax, int opsetVersion );

	// CNode methods' realizations
	void CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine ) override;
	void LabelTensorDims( const CTensorCache& tensors, CDimCache& dims ) override;
	void AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
		CNeoMLLinkCache& neoMLLinks, CDnn& dnn ) override;

private:
	// Softmax is applied along all the axes since this one (this one is included)
	int axis;

	CSoftmaxLayer::TNormalizationArea getArea( const CTensorShape& shape, const CTensorDim& dim );
};

} // namespace NeoOnnx
