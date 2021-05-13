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

#include "../Operator.h"

namespace NeoOnnx {

// Slice operator
class CSliceOperator : public CLayerOperator {
public:
	CSliceOperator( const onnx::NodeProto& slice, int opsetVersion );

	// COperator methods
	void AddLayers( const CObjectArray<const CTensorBase>& inputs,
		CDnn& dnn, CObjectArray<const CTensorBase>& outputs ) override;

	// CLayerOperator methods
	void UserInputMask( CUserInputMask& mask ) const override
		{ mask.Add( true ); mask.Add( false, InputCount() - 1 ); }

private:
	void getAxes( const CObjectArray<const CTensorBase>& inputs, CFastArray<int, 8>& axes ) const;
	void getStarts( const CObjectArray<const CTensorBase>& inputs, CFastArray<int, 8>& starts ) const;
	void getEnds( const CObjectArray<const CTensorBase>& inputs, CFastArray<int, 8>& ends ) const;
	void getSteps( const CObjectArray<const CTensorBase>& inputs, CFastArray<int, 8>& steps ) const;
	CPtr<const CUserTensor> sliceAxis( const CUserTensor& input, int axis, int start, int end, int step ) const;
	CPtr<const CUserTensor> prepareInputForSlice( const CUserTensor& input, int axis ) const;
};

} // namespace NeoOnnx
