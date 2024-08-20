/* Copyright © 2017-2023 ABBYY

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

// Forward declaration(s)
namespace NeoML {
class COnnxEltwiseLayer;
namespace optimization {
class CGraph;
} // namespace optimization
} // namespace NeoML

namespace NeoOnnx {

namespace optimization {

// Replaces the following construction
//
// *input*    CDataLayer(bias, single float)
//    |          |
//  COnnxEltwiseLayer(Sum or Sub)
//    |
//  ReLU      CDataLayer(slope, single float)
//    |          |
//  COnnxEltwiseLayer(Mul or Div)
//    |
// *output*
//
// with the next one:
//
//        *input*
//           |
// HardSigmoid(slope, bias)
//           |
//        *output*

class CHardSigmoidOptimizer final {
public:
	explicit CHardSigmoidOptimizer( NeoML::optimization::CGraph& graph ) :
		graph( graph )
	{}

	int Apply();

private:
	NeoML::optimization::CGraph& graph;

	// Checks if data layer is valid for CHardSigmoid conversion
	bool isValidDataLayer( CDataLayer& dataLayer, float& value ) const;
};

} // namespace optimization

} // namespace NeoOnnx
