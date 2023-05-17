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

#include <NeoML/Dnn/Optimization/Graph.h>

namespace NeoOnnx {

namespace optimization {

// Replaces the following construction
//
//       *input*
//          |
//  +-------+--------+
//  |                |
//  |   HardSigmoid(slope=1/6, bias=1/2)
//  |                |
// OnnxEltwiseLayer(Mul)
//          |
//       *output*
//
// with the hard swish (hswish) layer
class CHSwishOptimizer final {
public:
	explicit CHSwishOptimizer( NeoML::optimization::CGraph& graph ) :
		graph( graph )
	{}

	void Apply();

private:
	NeoML::optimization::CGraph& graph;

	bool isValidHardSigmoidLayer( CHardSigmoidLayer& hardSigmoidLayer,
		const NeoML::optimization::CLayerOutput<>& hSwishInputData ) const;
};

} // namespace optimization

} // namespace NeoOnnx
