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

// Forward declaration(s).
namespace onnx {
class NodeProto;
} // namespace onnx

namespace NeoOnnx {

class CReduceMeanNode : public CNode {
public:
	CReduceMeanNode( const onnx::NodeProto& reduceMean, CMap<CString, CInputInfo>& nodeOutputs );

	// CNode methods' realizations.
	void OnnxReshape() override;
	void MarkTensorDims() override;
	void AddLayers( CDnn& dnn ) override;

private:
	const int keepDims; // keep reduced dimensions (of size 1) or remove them.
	CArray<int> axes; // reduced axes.

	void add2dPoolingLayer( CDnn& dnn, int pooledDims );
};

} // namespace NeoOnnx
