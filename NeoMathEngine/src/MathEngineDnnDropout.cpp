/* Copyright © 2017-2024 ABBYY Production LLC

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
#include <NeoMathEngine/NeoMathEngine.h>
#include <CpuMathEngine.h>
#include <algorithm>

namespace NeoML {

static inline int getMaskSize( float rate, bool isSpatial, bool isBatchwise, const CBlobDesc& input )
{
	if( rate == 0 ) {
		return 0;
	}

	const int objectSize = isSpatial ? input.Channels() : input.ObjectSize();
	const int batchLength = isBatchwise ? input.ObjectCount() : input.BatchLength();
	const int batchWidth = input.ObjectCount() / batchLength;

	return batchWidth * objectSize;
}

static inline int getCacheSize(float rate, bool isSpatial, bool isBatchwise, const CBlobDesc& input)
{
	const int cacheSize = 64;
	const int maskSize = getMaskSize(rate, isSpatial, isBatchwise, input);
	if (isSpatial) {
		const int maskAlign = 4;
		const int blockSize = (cacheSize / input.Channels() + 1) * input.Channels() * maskAlign;
		return std::min(maskSize, blockSize);

	} else {
		return std::min(maskSize, cacheSize);
	}
}

CMaskDropoutDesc::CMaskDropoutDesc( IMathEngine& mathEngine, float rate, bool isSpatial, bool isBatchwise,
		const CBlobDesc& input, const CBlobDesc& output, int seed ) :
	Input( input ),
	Output( output ),
	ForwardRate( 1.f - rate ),
	IsSpatial( isSpatial ),
	IsBatchwise( isBatchwise ),
	Mask( mathEngine, getMaskSize( rate, isSpatial, isBatchwise, input ) )
{
	ASSERT_EXPR( rate >= 0.f && rate < 1.f );

	if( rate != 0 ) {
		mathEngine.VectorFillBernoulli( Mask.GetHandle(), ForwardRate, Mask.Size(), 1.f / ForwardRate, seed );
	}
}

CSeedDropoutDesc::CSeedDropoutDesc( float rate, bool isSpatial, bool isBatchwise,
		const CBlobDesc& input, const CBlobDesc& output, int seed ) :
	Input( input ),
	Output( output ),
	ForwardRate( 1.f - rate ),
	IsSpatial( isSpatial ),
	IsBatchwise( isBatchwise ),
	seed(seed),
	threshold( (unsigned int)((double)ForwardRate* UINT_MAX) ),
	value( 1.f / ForwardRate ),
	Mask( mathEngine, getCacheSize(rate, isSpatial, isBatchwise, input) )
{

	ASSERT_EXPR( rate >= 0.f && rate < 1.f );
}

} // namespace NeoML
