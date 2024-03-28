/* Copyright © 2017-2024 ABBYY

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

#include <common.h>
#pragma hdrstop

#include <MathEngineDnnDropout.h>

namespace NeoML {

CSeedDropoutDesc::CSeedDropoutDesc(float rate, bool isSpatial, bool isBatchwise, const CBlobDesc& input,
	const CBlobDesc& output, int seed) :
		Input(input),
		Output(output),
		ForwardRate(1.f - rate),
		IsSpatial(isSpatial),
		IsBatchwise(isBatchwise),
		Value(1.f / (1.f - rate)),
		Seed( seed ),
		Threshold((unsigned int)(ForwardRate* UINT_MAX))
{
	ASSERT_EXPR(rate >= 0.f && rate < 1.f);
}

} // namespace NeoML
