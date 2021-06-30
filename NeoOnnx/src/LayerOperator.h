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

	// Default implementation which calls protected ProcessTensors with default input mask (only first input)
	// See comment to the protected version for more details
	void ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;

protected:
	CLayerOperator( const onnx::NodeProto& node, int opsetVersion ) : COperator( node, opsetVersion ) {}

	// Default implementation which works in the next way:
	// If output tensors' data can't be calculated it just adds corresponding layers to the dnn
	// Otherwise it creates another CDnn, adds layers to the new CDnn
	// and uses this internal network to calculate output tensors' data
	// inputMask indicates whether i'th input should be a CUserInput of internalDnn (instead of CDataTensor)
	// e.g. for CConvOperator only first input must be a CUserTensor (filters and free terms should remain as CDataTensor)
	// on the other hand for CConcatOperator each one of the inputs must be a CUserTensor
	void ProcessTensors( const CUserInputMask& inputMask, const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const;

	// Virtual methods

	// Adds layers required which are imitating this operator to the dnn
	// and puts corresponding CUserTensor's to the outputs
	virtual void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const = 0;

private:
	bool canCalculateOutput( const CTensorArray& inputs ) const;
	void addInternalDnnSources( const CUserInputMask& inputMask, const CTensorArray& inputs,
		CTensorArray& internalInputs, CDnn& internalDnn ) const;
	void addInternalDnnSinks( const CTensorArray& internalOutputs, CArray<CSinkLayer*>& sinks, CDnn& internalDnn ) const;
	void extractOutputs( const CTensorArray& internalOutputs, const CArray<CSinkLayer*>& sinks, CTensorArray& outputs ) const;
};

} // namespace NeoOnnx
