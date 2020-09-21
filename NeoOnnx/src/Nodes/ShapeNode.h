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

// Shape node in onnx graph
class CShapeNode : public COpNode {
public:
	CShapeNode( int nodeIndex, const onnx::NodeProto& shape, int opsetVersion );

	// CNode methods' realizations
	void CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine ) override;
	void MarkTensorDims( const CTensorCache& tensors, CDimCache& dims ) override {}
	void AddLayers( const CGraph&, const CTensorCache&, const CDimCache&, CNeoMLLinkCache&, CDnn& ) override {}
};

} // namespace NeoOnnx
