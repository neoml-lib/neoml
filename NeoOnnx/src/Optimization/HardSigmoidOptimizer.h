/* Copyright © 2017-2023 ABBYY Production LLC

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

#include "Optimizer.h"

// Forward declaration(s)
namespace NeoML {
class COnnxEltwiseLayer;
} // namespace NeoML

namespace NeoOnnx {

namespace optimization {

// Forward declaration(s)
class CGraph;

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

class CHardSigmoidOptimizer : public IOptimizer {
public:
	explicit CHardSigmoidOptimizer( CGraph& graph ) :
		graph( graph )
	{
	}

	void Apply() override;

private:
	CGraph& graph;

	bool isValidDataLayer( CDataLayer& dataLayer, float& value ) const;
};

} // namespace optimization

} // namespace NeoOnnx
