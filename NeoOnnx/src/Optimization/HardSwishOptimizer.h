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
#include "DnnGraphWrapper.h"
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

namespace NeoOnnx {

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

class CHardSwishOptimizer : public IOptimizer {
public:
	explicit CHardSwishOptimizer( CDnn& dnn ) :
		IOptimizer( dnn, nullptr ),
		graph( dnn )
	{
	}

	void Apply() override;

private:
	CDnnGraphWrapper graph;

	bool isValidHardSwish( const COnnxEltwiseLayer& mulLayer, CHardSigmoidLayer*& hardSigmoidLayer,
		CDnnGraphLink& hardSwishInput ) const;
	bool isValidHardSigmoidLayer( const CHardSigmoidLayer& hardSigmoidLayer,
		const CDnnGraphLink& hardSwishInput ) const;
};

} // namespace NeoOnnx
