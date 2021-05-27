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
	// Fills the array with bools where true means that index'th input
	// is expected to be provided by user and false otherwise
	// Used in COperator::CalculateOutput
	virtual void UserInputMask( CUserInputMask& mask ) const = 0;

	// COperator's interface

	// Default implementation which works for the most of the derivatives
	bool CanCalculateOutput( const CObjectArray<const CTensorBase>& inputs ) const final;

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
