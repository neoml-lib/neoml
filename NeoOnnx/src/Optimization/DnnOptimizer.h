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

#include "Graph.h"
#include "HardSigmoidOptimizer.h"
#include "HardSwishOptimizer.h"

namespace NeoOnnx {

namespace optimization {

// CDnnOptimizer provides a reconstruction of the CDnn.
// NOTE: The underlying CDnn would be changed
class CDnnOptimizer final {
public:
	explicit CDnnOptimizer( CDnn& dnn ) :
		graph( dnn )
	{
	}
	CDnnOptimizer( CDnnOptimizer&& ) = delete;
	CDnnOptimizer( const CDnnOptimizer& ) = delete;

	void Optimize();

private:
	CGraph graph;
};

inline void CDnnOptimizer::Optimize()
{
	CHardSigmoidOptimizer( graph ).Apply();
	CHardSwishOptimizer( graph ).Apply();
}

} // namespace optimization

} // namespace NeoOnnx

