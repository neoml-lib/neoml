/* Copyright © 2017-2021 ABBYY Production LLC

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

#include "PyLayer.h"

class CPyBaseSplitLayer : public CPyLayer {
public:
	CPyBaseSplitLayer( CBaseSplitLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::array GetOutputCounts() const
	{
		const auto& fineCounts = Layer<CBaseSplitLayer>()->GetOutputCounts();

		py::array_t<int, py::array::c_style> counts( fineCounts.Size() );
		auto countsData = counts.mutable_unchecked<>();

		for( int i = 0; i < fineCounts.Size(); ++i ) {
			countsData( i ) = fineCounts[i];
		}

		return counts;
	}

	void SetOutputCounts( py::array counts )
	{
		NeoAssert( counts.ndim() == 1 );
		NeoAssert( counts.dtype().is( py::dtype::of<int>() ) );

		CArray<int> fineCounts;
		fineCounts.SetSize( static_cast<int>(counts.size()) );
		for( int i = 0; i < fineCounts.Size(); i++ ) {
			fineCounts[i] = static_cast<const int*>(counts.data())[i];
		}
		Layer<CBaseSplitLayer>()->SetOutputCounts( fineCounts );
	}
};

void InitializeSplitLayer( py::module& m );