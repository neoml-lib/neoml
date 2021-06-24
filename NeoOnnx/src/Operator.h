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

#include "OperatorAttributes.h"
#include "TensorLayout.h"
#include "Tensor.h"

// Forward declaration(s)
namespace onnx {
class NodeProto;
} // namespace onnx

namespace NeoML {
class IMathEngine;
} // namespace NeoML

namespace NeoOnnx {

// Opset versioning support
const int MaxOpsetVersion = 12;

// onnx operator
class COperator {
public:
	virtual ~COperator() = default;

	COperator( const COperator& other ) = delete;
	COperator& operator= ( const COperator& other ) = delete;

	// Static methods

	// Fabric method
	// Creates COperator's derivative for the given onnx proto
	static COperator* CreateOperator( const onnx::NodeProto& onnxNode, int opsetVersion );

	// Returns true if operatorType is supported by NeoOnnx
	static bool IsSupportedOperator( const CString& operatorType );

	// Properties

	// Operator's name
	const CString& Name() const { return name; }
	// Operator's type
	const CString& Type() const { return type; }
	// Number of inputs
	int InputCount() const { return inputNames.Size(); }
	// Name of the index'th input
	const CString& InputName( int index ) const;
	// Number of outputs
	int OutputCount() const { return outputNames.Size(); }
	// Name of the index'th output
	const CString& OutputName( int outputIndex ) const;

	// Virtual methods

	// Puts output tensors to the output array
	// If data can be calculated the output tensors will be of CDataTensor type
	// Otherwise output tensors will be of CUserTensor type and corresponding layers will be added to dnn
	virtual void GetOutputTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const = 0;

protected:
	COperator( const onnx::NodeProto& node, int opsetVersion );

	// Opset version
	const int OpsetVersion;
	// Attributes of this operator
	const COperatorAttributes Attributes;

private:
	// Operator name
	CString name;
	// Operator type
	const CString type;
	// Input names
	CArray<CString> inputNames;
	// Output names
	CArray<CString> outputNames;
};

} // namespace NeoOnnx
