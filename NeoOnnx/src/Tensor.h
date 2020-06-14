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

namespace NeoOnnx {

// Forward declaration(s).
class CGraph;

// Type of data of node input in an ONNX graph.
enum TTensorType {
	TT_ConstantTensor, // tensor with values independent of graph inputs
	TT_DataTensor, // tensor with values dependent on graph inputs

	TT_Count
};

// Match between ONNX tensor axes and NeoML dimensions.
typedef CFastArray<TBlobDim, 8> CTensorDim;

// Tensor shape (in onnx notation).
typedef CFastArray<int, 8> CTensorShape;

// ONNX tensor.
class CTensor {
public:
	explicit CTensor( TTensorType type, const CTensorShape& shape = {}, CDnnBlob* data = nullptr );
	CTensor( const CTensor& other );
	CTensor() = default;

	// Type of node input.
	TTensorType GetType() const { return type; }

	// Shape of node input. Available only if data type is shape or tensor.
	const CTensorShape& GetShape() const;

	// Data of node input. Available only if data type in constant tensor.
	const CDnnBlob* GetData() const;
	CDnnBlob* GetData();

	// Sets NeoML dimensions of the tensor.
	// Returns true if there is no conflicts.
	bool SetTensorDim( const CTensorDim& supposedDim );

	// Gets NeoML dimensions of the tensor.
	const CTensorDim& GetTensorDim() const { return tensorDim; }

private:
	TTensorType type; // tensor type.
	CTensorShape shape; // tensor shape.
	CTensorDim tensorDim; // tensor NeoML dimension.
	CPtr<CDnnBlob> data; // tensor data (if can be calculated).
};

} // namespace NeoOnnx
