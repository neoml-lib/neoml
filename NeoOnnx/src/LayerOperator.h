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

#include "Operator.h"

namespace NeoOnnx {

// Determines whether index'th input is expected to be provided by user or not
typedef CDynamicBitSet<8> CUserInputMask;

// Operator, which can be emulated by NeoML layers
// Provides default implementation of one of CNode's methods and adds new method to the interface
class CLayerOperator : public COperator {
public:
	// COperator's interface

	// Default implementation which works in the next way:
	// If output tensors' data can't be calculated it just adds corresponding layers to the dnn
	// Otherwise it creates another CDnn, adds layers to the new CDnn
	// and uses this internal network to calculate output tensors' data
	void GetOutputTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const final;

protected:
	CLayerOperator( const onnx::NodeProto& node, int opsetVersion ) : COperator( node, opsetVersion ) {}

	// Virtual methods

	// Fills the mask with bools where true means that index'th input
	// is expected to be provided by user for the correct work of CLayerOperator::AddLayers
	virtual void UserInputMask( CUserInputMask& mask ) const = 0;

	// Adds layers required which are imitating this operator to the dnn
	// and puts corresponding CUserTensor's to the outputs
	virtual void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const = 0;

private:
	bool canCalculateOutput( const CTensorArray& inputs ) const;
	void addInternalDnnSources( const CTensorArray& inputs, CTensorArray& internalInputs, CDnn& internalDnn ) const;
	void addInternalDnnSinks( const CTensorArray& internalOutputs, CArray<CSinkLayer*>& sinks, CDnn& internalDnn ) const;
	void extractOutputs( const CTensorArray& internalOutputs, const CArray<CSinkLayer*>& sinks, CTensorArray& outputs ) const;
};

} // namespace NeoOnnx
