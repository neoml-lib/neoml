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

#include "OpNodeAttributes.h"
#include "TensorLayout.h"
#include "Tensor.h"
#include "NeoMLLink.h"
#include "NeoOnnxCheck.h"

#include <onnx.pb.h>

// Forward declaration(s)
namespace NeoML {
class IMathEngine;
} // namespace NeoML

namespace NeoOnnx {

// Node in the NeoOnnx graph
class CNode {
public:
	virtual ~CNode() = default;

	// Node name
	const CString& Name() const { return name; }
	// Number of inputs
	int InputCount() const { return inputNames.Size(); }
	// Name of the index'th input
	const CString& InputName( int index ) const;
	// Number of outputs
	int OutputCount() const { return outputNames.Size(); }
	// Name of the index'th output
	const CString& OutputName( int outputIndex ) const;

	// Virtual methods

	// Returns true if node has all the data required for computing output during generation
	// This method has default implementation which works for the most of derivatives
	virtual bool CanCalculateOutput( const CObjectArray<const CTensorBase>& inputs ) const;

	// Adds required layers to dnn and puts corresponding tensors to the outputs
	// Called if operator output depends on the data, provided by user
	virtual void AddLayers( const CObjectArray<const CTensorBase>& inputs,
		CObjectArray<const CTensorBase>& outputs, CDnn& dnn ) const = 0;

	// Calculates the result of the operations
	// Called if operator's output can be calculated during network conversion 
	// (which means that tensor's data is independent of user input)
	virtual void CalculateOutput( const CObjectArray<const CTensorBase>& inputs,
		CObjectArray<const CTensorBase>& outputs, IMathEngine& mathEngine ) const = 0;

protected:
	CNode( const CString& name, const CArray<CString>& inputs, const CArray<CString>& outputs );
	CNode( const CString& name, const ::google::protobuf::RepeatedPtrField<std::string>& inputs,
		const ::google::protobuf::RepeatedPtrField<std::string>& outputs );

private:
	// Node name
	CString name;
	// Input names
	CArray<CString> inputNames;
	// Output names
	CArray<CString> outputNames;
};

//--------------------------------------------------------------------------------------------------------------------
// Opset versioning support
const int MaxOpsetVersion = 12;

//--------------------------------------------------------------------------------------------------------------------
// 
typedef CFastArray<bool, 8> CUserInputMask;

//---------------------------------------------------------------------------------------------------------------------
// Operator node
class COpNode : public CNode {
public:
	~COpNode() override = default;

	// Default implementation which imitates pre-calculation in the following way:
	// 1. Creates small CDnn and creates appropriate sources
	// 2. Calling AddLayers CNode's interface method for that internalDnn
	// 3. Running this CDnn and extracting the results
	void CalculateOutput( const CObjectArray<const CTensorBase>& inputs,
		CObjectArray<const CTensorBase>& outputs, IMathEngine& mathEngine ) const override;

	// Fills the array with bools where true means that index'th input
	// is expected to be provided by user and false otherwise
	// Used in COpNode::CalculateOutput
	virtual void UserInputMask( CUserInputMask& mask ) const = 0;

	// Fabric method. Creates CNode's derivative for given onnx node
	static COpNode* CreateOpNode( const onnx::NodeProto& onnxNode, int opsetVersion );

	// Returns true if operator opType is supported by NeoOnnx
	static bool IsSupportedOperator( const CString& opType );

protected:
	COpNode( const onnx::NodeProto& node, int opsetVersion );

	const int OpsetVersion; // Opset version
	const COpNodeAttributes Attributes; // Attributes of this node
	const onnx::NodeProto OnnxNode; // Reference to onnx node (used for diagnostics)

private:
	void addInternalDnnSources( const CObjectArray<const CTensorBase>& inputs,
		CObjectArray<const CTensorBase>& internalInputs, CDnn& internalDnn ) const;
	void addInternalDnnSinks( const CObjectArray<const CTensorBase>& internalOutputs,
		CArray<CSinkLayer*>& sinks, CDnn& internalDnn ) const;
	void extractOutputs( const CObjectArray<const CTensorBase>& internalOutputs,
		const CArray<CSinkLayer*>& sinks, CObjectArray<const CTensorBase>& outputs ) const;
};

} // namespace NeoOnnx
