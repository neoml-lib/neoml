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

#include <type_traits>
#include "Optimization/Graph.h"
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>
#include "../TensorUtils.h"

namespace NeoOnnx {

namespace optimization {

//  Layer Normalization will fuse ObjectLayerNormalization into one layer:
// 
//  (x - mean(x, axis)) / sqrt(var(x, axis)) * scale + bias  , where 'x' is the input and var(x) = mean((x-mean)^2).
// 
//  +---------------------+
//  |                     |
//  |                     v
//  X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
//                        |                                               ^
//                        |                                               |
//                        +-----------------------------------------------+
//  It also handles cases of duplicated sub layers exported from older version of PyTorch :
//  +---------------------+
//  |                     v
//  |          +-------> Sub ---------------------------------------------+
//  |          |                                                          |
//  |          |                                                          v
//  X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
//  |                     ^
//  |                     |
//  +---------------------+
// 
//  In recent pytorch, Cast layers may be inserted before Pow to ensure that both inputs 'base' and 'power' are the same type
//  due to restriction in older opsets. Therefore, Layer Normalization will also handle the case below :
//  +---------------------+
//  |                     |
//  |                     v
//  X --> ReduceMean --> Sub --> Cast --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
//                        |                                                        ^
//                        |                                                        |
//                        +--------------------------------------------------------+
//  +---------------------+       Cast
//  |                     |        |
//  |                     v        v
//  X --> ReduceMean --> Sub -->  Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
//                        |                                                ^
//                        |                                                |
//                        +------------------------------------------------+
// 
//  When using Apex O2, a Cast layer may be inserted between Div and Mul, Layer Normalization will also handle the case below:
//  +---------------------+
//  |                     |
//  |                     v
//  X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast --> Mul --> Add
//                        |                                               ^
//                        |                                               |
//                        +-----------------------------------------------+
//  OR
//           +---------------------+
//           |                     |
//           |                     v
//  X --> Cast --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast --> Mul --> Add
//                                 |                                               ^
//                                 |                                               |
//                                 +-----------------------------------------------+
//  Logically since LayerNormalization supports input and scale/bias in different data types, and during the kernel execution,
//  data are casted to float/double to calculate for precision, so if there is any Cast Ops in the sub-graph, we can remove it.
//  Such Cast Op can be the input of the sub-graph, or an Cast Op between the Div and Mul layers.
class CLayerNormFusionOptimizer final {
public:
	explicit CLayerNormFusionOptimizer( CGraph& graph ) : graph(graph) {}
	CLayerNormFusionOptimizer( const CLayerNormFusionOptimizer& ) = delete;
	CLayerNormFusionOptimizer( CLayerNormFusionOptimizer&& ) = delete;

	int Apply();

private:
	CGraph& graph;
	struct CLayoutChange;

	CLayerOutput<> selectLayoutChange( CBaseLayer& inputLayer, int inputIndex, CLayoutChange& change ) const;
	bool isValidDataLayer( const CDataLayer& dataLayer, TBlobType blobType, int blobSize = NotFound ) const;
	bool isValidCastLayer( const CCastLayer& castLayer ) const;
	// Checks if COnnxEltwiseLayer is valid for CLayerNormFusionOptimizer conversion
	bool isValidArithmeticLayer( const COnnxEltwiseLayer& layer, COnnxEltwiseLayer::TOperation operation ) const
	{ return layer.GetOperation() == operation && graph.GetInputCount( layer ) == 2 && graph.GetOutputCount( layer ) == 1; }
	// Checks if CPowerLayer is valid for CLayerNormFusionOptimizer conversion
	bool isValidPowerLayer( const CPowerLayer& powLayer, float exponent ) const
	{ return powLayer.GetExponent() == exponent && graph.GetInputCount( powLayer ) == 1 && graph.GetOutputCount( powLayer ) == 1; }
	CLayerOutput<> changeLayout( const CLayerOutput<>& inputData, const CTensorLayout& inputLayout,
		const ITensorLayoutValidator& validator, CTensorLayout& outputLayout );
};

} // namespace optimization

} // namespace NeoOnnx

