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

// Unsqueeze operator
class CUnsqueezeOperator : public CLayerOperator {
public:
	CUnsqueezeOperator( const onnx::NodeProto& unsqueeze, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;
	void UserInputMask( CUserInputMask& mask ) const override { mask |= 0; }

private:
	void getAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const;
	void calcOutputShape( const CTensorShape& inputShape, const CFastArray<int, 8>& axes, CTensorShape& outputShape ) const;
	CTensorLayout calcOutputLayout( const CTensorLayout& inputLayout, const CFastArray<int, 8>& axes ) const;
};

} // namespace NeoOnnx
