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

#include "onnx.pb.h"

#include "TensorLayout.h"
#include "Tensor.h"

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
	// Operator's type ('Conv', 'Pool' etc.)
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

	// Returns true if operator has attribute with the given name
	bool HasAttribute( const CString& name ) const;

	// Gets attribute value
	// Returns false if there is no attribute with the given name
	template<class T>
	bool GetAttribute( const CString& name, T& value ) const;

	void DebugPrintInputs( const CTensorArray& inputs ) const
	{
		printf( "Operator: %s(%s)\n", static_cast<const char*>( Name() ), static_cast<const char*>( Type() ) );
		for( int i = 0; i < inputs.Size(); ++i ) {
			::printf( "  input[%d]:", i );
			if( inputs[i] == nullptr ) {
				::printf( "  NULL\n" );
				continue;
			}
			::printf( inputs[i]->IsCalculated() ? "  DATA " : "  USER " );
			const CTensorShape& inputShape = inputs[i]->Shape();
			for( int j = 0 ;j < inputShape.Size(); ++j ) {
				::printf( " %d", inputShape[j] );
			}
			::printf( "\n" );
		}
		printf( "\n" );
	}

	// Opset version
	const int OpsetVersion;

private:
	// Operator name
	const CString name;
	// Operator type
	const CString type;
	// Operator attributes
	CMap<CString, const onnx::AttributeProto&> attributes;
	// Input names
	CArray<CString> inputNames;
	// Output names
	CArray<CString> outputNames;

	template<class T>
	void getAttributeValue( const onnx::AttributeProto& attribute, T& value ) const;
};

} // namespace NeoOnnx

#include "OperatorAttributeGetters.inl"
