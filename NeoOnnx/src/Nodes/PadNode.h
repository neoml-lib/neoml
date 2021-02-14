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

#include "../Node.h"

namespace NeoOnnx {

// Auxiliary functions which are used in all nodes that support padding (Pad/Pool/Conv)

// Calculates padding size if autoPad is SAME_*
void CalculatePadding( const CString& autoPad, const CTensorShape& kernelShape, CFastArray<int, 8>& pads );

// Pads tensor with user-dependent data
CPtr<const CUserTensor> PadUserTensor( const CUserTensor& inputs, const CFastArray<int, 8>& pads, float padValue );

//---------------------------------------------------------------------------------------------------------------------

// Pad operator graph node
class CPadNode : public COpNode {
public:
	CPadNode( const onnx::NodeProto& pad, int opsetVersion );

	// CNode methods
	void AddLayers( const CObjectArray<const CTensorBase>& inputs,
		CObjectArray<const CTensorBase>& outputs, CDnn& dnn ) override;

	// COpNode methods
	void UserInputMask( CUserInputMask& mask ) const override
		{ mask.Add( true ); mask.Add( false, InputCount() - 1 ); }

private:
	CString mode; // Pad mode
	float value; // Pad value
	CFastArray<int, 8> pads; // Pad sizes
};

} // namespace NeoOnnx
