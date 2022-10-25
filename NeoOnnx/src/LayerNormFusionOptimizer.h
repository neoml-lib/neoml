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

#include "Optimizer.h"

namespace NeoOnnx {

//------------------------------------------------------------------------------------------------------------
/**
 *  Layer Normalization will fuse ObjectLayerNormalization into one layer:
 *
 *  (x - mean(x, axis)) / sqrt(var(x, axis)) * scale + bias  , where 'x' is the input and var(x) = mean((x-mean)^2).
 *
 *  +---------------------+
 *  |                     |
 *  |                     v
 *  X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
 *                        |                                               ^
 *                        |                                               |
 *                        +-----------------------------------------------+
 *  It also handles cases of duplicated sub layers exported from older version of PyTorch :
 *  +---------------------+
 *  |                     v
 *  |          +-------> Sub ---------------------------------------------+
 *  |          |                                                          |
 *  |          |                                                          v
 *  X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
 *  |                     ^
 *  |                     |
 *  +---------------------+
 *
 *  In recent pytorch, Cast layers may be inserted before Pow to ensure that both inputs 'base' and 'power' are the same type
 *  due to restriction in older opsets. Therefore, Layer Normalization will also handle the case below :
 *  +---------------------+
 *  |                     |
 *  |                     v
 *  X --> ReduceMean --> Sub --> Cast --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
 *                        |                                                        ^
 *                        |                                                        |
 *                        +--------------------------------------------------------+
 *  +---------------------+       Cast
 *  |                     |        |
 *  |                     v        v
 *  X --> ReduceMean --> Sub -->  Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
 *                        |                                                ^
 *                        |                                                |
 *                        +------------------------------------------------+
 *
 *  When using Apex O2, a Cast layer may be inserted between Div and Mul, Layer Normalization will also handle the case below:
 *  +---------------------+
 *  |                     |
 *  |                     v
 *  X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast --> Mul --> Add
 *                        |                                               ^
 *                        |                                               |
 *                        +-----------------------------------------------+
 *
 *  OR
 *
 *           +---------------------+
 *           |                     |
 *           |                     v
 *  X --> Cast --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast --> Mul --> Add
 *                                 |                                               ^
 *                                 |                                               |
 *                                 +-----------------------------------------------+
 *
 *  Logically since LayerNormalization supports input and scale/bias in different data types, and during the kernel execution,
 *  data are casted to float/double to calculate for precision, so if there is any Cast Ops in the sub-graph, we can remove it.
 *  Such Cast Op can be the input of the sub-graph, or an Cast Op between the Div and Mul layers.
 **/
class CLayerNormFusionOptimizer final : public IOptimizer {
public:
	explicit CLayerNormFusionOptimizer( CDnn& graph ) : IOptimizer( graph ) {}

	void Apply() override;
};

} // namespace NeoOnnx

