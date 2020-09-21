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

#include "Tensor.h"
#include "GraphCache.h"

// Forward declaration(s)
namespace onnx {
class NodeProto;
} // namespace onnx

namespace NeoOnnx {

// Calculates the padding of the operation with 'attributes' for the last 'kernelShape.Size()' dimensions of the 'inputShape'
void CalculatePadding( const CString& autoPad, const CTensorShape& inputShape,
	const CTensorShape& kernelShape, CFastArray<int, 8>& pads, const onnx::NodeProto& onnxNode );

// Repacks weights from channel-frst to channel-last if node if flatten operator
// Required for Gemm, LSTM etc
// Returns the pointer to the same blob if repack isn't needed
CPtr<CDnnBlob> RepackWeightIfFlattened( const CNode* node, const CTensorCache& tensors, const CDimCache& dims, CDnnBlob* weight );

} // namespace NeoOnnx
