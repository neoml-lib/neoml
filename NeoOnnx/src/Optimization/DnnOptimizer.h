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
#include "HardSigmoidOptimizer.h"
#include "HSwishOptimizer.h"
#include "LayerNormFusionOptimizer.h"
#include "SqueezeAndExciteOptimizer.h"

namespace NeoOnnx {

namespace optimization {

// CDnnOptimizer provides a reconstruction of the CDnn.
// NOTE: The underlying CDnn would be changed
class CDnnOptimizer final {
public:
	explicit CDnnOptimizer( CDnn& dnn ) :
		graph( dnn )
	{}
	CDnnOptimizer( CDnnOptimizer&& ) = delete;
	CDnnOptimizer( const CDnnOptimizer& ) = delete;

	void Optimize();

private:
	NeoML::optimization::CGraph graph;
};

inline void CDnnOptimizer::Optimize()
{
	CHardSigmoidOptimizer( graph ).Apply();
	CHSwishOptimizer( graph ).Apply();
	CSqueezeAndExciteOptimizer( graph ).Apply();
	//CLayerNormFusionOptimizer( graph, dnn ).Apply();
}

} // namespace optimization

} // namespace NeoOnnx

