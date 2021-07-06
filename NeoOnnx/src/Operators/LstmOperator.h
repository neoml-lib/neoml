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

#include "../LayerOperator.h"

namespace NeoOnnx {

// Lstm operator
class CLstmOperator : public CLayerOperator {
public:
	CLstmOperator( const onnx::NodeProto& lstm, int opsetVersion );

	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;

	// COperator methods
	void UserInputMask( CUserInputMask& mask ) const override { mask |= 0; }

private:
	// LSTM's direction ("forward", "backward" or "bidirectional")
	const CString direction;
	// Size of hidden state vector
	const int hiddenSize;

	CPtr<CDnnBlob> reorderGates( CPtr<CDnnBlob> weights, TBlobDim dim ) const;
};

} // namespace NeoOnnx
