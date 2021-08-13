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

// Operator, which can be emulated by NeoML layers
// Provides default implementation of one of CNode's methods and adds new method to the interface
class CLayerOperator : public COperator {
public:
	// COperator's interface

	// Default implementation which works in the next way:
	// If output tensors' data can't be calculated it just adds corresponding layers to the dnn
	// Otherwise it creates another internal CDnn, adds layers to this new CDnn
	// and uses this internal network to calculate output tensors' data
	void ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;

protected:
	CLayerOperator( const onnx::NodeProto& node, int opsetVersion ) : COperator( node, opsetVersion ) {}

	// Virtual methods

	// Adds layers required which are imitating this operator to the dnn
	// and puts corresponding CUserTensor's to the outputs
	virtual void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const = 0;
};

} // namespace NeoOnnx
