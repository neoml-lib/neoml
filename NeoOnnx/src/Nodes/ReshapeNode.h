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

class CReshapeNode : public COpNode {
public:
	CReshapeNode( int nodeIndex, const onnx::NodeProto& node, int opsetVersion );

	// CNode methods' realizations.
	void CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine ) override;
	void MarkTensorDims( const CTensorCache& tensors, CDimCache& dims ) override {}
	void AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
		CNeoMLLinkCache& neoMLLinks, CDnn& dnn ) override;

private:
	bool hasFixedShape; // Does this op has fixed shape for any of dimensions (shape[i] > 0)
	bool hasRemainder; // Does this op has a remainder dimension (shape[i] == -1)

	CArray<int> shape; // Shape input value
};

} // namespace NeoOnnx
