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

// Gemm operator
class CGemmOperator : public CLayerOperator {
public:
	CGemmOperator( const onnx::NodeProto& node, int opsetVersion );

protected:
	// CLayerOperator methods
	void AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const override;

private:
	// In onnx Gemm is implemented like
	//     Y = alpha * A' * B' + beta * C
	// where
	//     A' is equal to A if transA == 0 or transpose(A) otherwise
	//     B' is equal to B if transB == 0 or transpose(B) otherwise
	//     C is optional matrix

	// Values from formula above
	float alpha;
	float beta;
	int transA;
	int transB;
};

} // namespace NeoOnnx
