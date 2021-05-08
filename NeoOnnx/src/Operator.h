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

#include <onnx.pb.h>

// Forward declaration(s)
namespace NeoML {
class IMathEngine;
} // namespace NeoML

namespace NeoOnnx {

// Opset versioning support
const int MaxOpsetVersion = 12;

// onnx operator
// This class adds some operators-only features support (type, attributes, opsetVersion)
// and fabric methods for operators
// Doesn't affect interfaces in any way
class COperator {
public:
	virtual ~COperator() = default;

	// Operator name
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

	// Returns true if operator has all the data required for computing output during generation
	// This method has default implementation which works for the most of derivatives
	virtual bool CanCalculateOutput( const CObjectArray<const CTensorBase>& inputs ) const;

	// Adds required layers to dnn and puts corresponding tensors to the outputs
	// Called if operator output depends on the data, provided by user
	virtual void AddLayers( const CObjectArray<const CTensorBase>& inputs,
		CDnn& dnn, CObjectArray<const CTensorBase>& outputs ) = 0;

	// Calculates the result of the operations
	// Called if operator's output can be calculated during network conversion 
	// (which means that tensor's data is independent of user input)
	virtual void CalculateOutput( const CObjectArray<const CTensorBase>& inputs,
		IMathEngine& mathEngine, CObjectArray<const CTensorBase>& outputs ) = 0;

	// Fabric method
	// Creates COperator's derivative for the given onnx proto
	static COperator* CreateOperator( const onnx::NodeProto& onnxNode, int opsetVersion );

	// Returns true if operatorType is supported by NeoOnnx
	static bool IsSupportedOperator( const CString& operatorType );

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

//---------------------------------------------------------------------------------------------------------------------

// Determines whether index'th input is expected to be provided by user or not
typedef CFastArray<bool, 8> CUserInputMask;

// Operator, which can be emulated by NeoML layers
// Provides default implementation of one of CNode's methods and adds new method to the interface
class CLayerOperator : public COperator {
public:
	// Fills the array with bools where true means that index'th input
	// is expected to be provided by user and false otherwise
	// Used in COperator::CalculateOutput
	virtual void UserInputMask( CUserInputMask& mask ) const = 0;

	// COperator's interface
	// Default implementation which imitates pre-calculation in the following way:
	// 1. Creates small CDnn and creates appropriate sources
	// 2. Calling AddLayers COperator's interface method for that internalDnn
	// 3. Running this CDnn and extracting the results
	// In future final may be removed (for optimization purposes)
	void CalculateOutput( const CObjectArray<const CTensorBase>& inputs,
		IMathEngine& mathEngine, CObjectArray<const CTensorBase>& outputs ) final;

	// COperator::AddLayers must be defined by derivatives

protected:
	CLayerOperator( const onnx::NodeProto& node, int opsetVersion )
		: COperator( node, opsetVersion ) {}

private:
	void addInternalDnnSources( const CObjectArray<const CTensorBase>& inputs,
		CObjectArray<const CTensorBase>& internalInputs, CDnn& internalDnn ) const;
	void addInternalDnnSinks( const CObjectArray<const CTensorBase>& internalOutputs,
		CArray<CSinkLayer*>& sinks, CDnn& internalDnn ) const;
	void extractOutputs( const CObjectArray<const CTensorBase>& internalOutputs,
		const CArray<CSinkLayer*>& sinks, CObjectArray<const CTensorBase>& outputs ) const;
};

} // namespace NeoOnnx
