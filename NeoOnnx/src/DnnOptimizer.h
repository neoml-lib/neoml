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
#include "LayerNormFusionOptimizer.h"

namespace NeoOnnx {

// CDnnOptimizer provides a reconstruction of the CDnn.
// NOTE: The underlying CDnn would be changed
class CDnnOptimizer final : public IObject {
public:
	explicit CDnnOptimizer( CDnn& graph ) :
		optimizers{ CPtr<IOptimizer>( new CLayerNormFusionOptimizer( graph ) ) }
	{}
	CDnnOptimizer( CDnnOptimizer&& ) = delete;
	CDnnOptimizer( const CDnnOptimizer& ) = delete;

	void Optimize();

private:
	CArray<CPtr<IOptimizer>> optimizers{};
};

inline void CDnnOptimizer::Optimize()
{
	for( int i = 0; i < optimizers.Size(); ++i ) {
		optimizers[i]->Apply();
	}
}

} // namespace NeoOnnx

