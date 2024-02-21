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

#pragma once

#include <NeoMathEngine/MathEngineDropout.h>
#include <NeoMathEngine/CrtAllocatedObject.h>

namespace NeoML {

static inline void updateDesc(CBlobDesc*& destDesc, const CBlobDesc* srcDesc) {
	if(destDesc == nullptr) {
		destDesc = new CBlobDesc(*srcDesc);
	} else {
		*destDesc = *srcDesc;
	}
}

static inline int getMaskSize(bool isSpatial, bool isBatchwise, const CBlobDesc& input)
{
	const int objectSize = isSpatial ? input.Channels() : input.ObjectSize();
	const int batchLength = isBatchwise ? input.ObjectCount() : input.BatchLength();
	const int batchWidth = input.ObjectCount() / batchLength;

	return batchWidth * objectSize;
}

// Dropout descriptor containing whole mask
struct CMaskDropoutDesc : public CBaseDropoutDesc {};

// Dropout descriptor containing fixed memory for generating mask parts iteratively
struct CSeedDropoutDesc : public CBaseDropoutDesc {
	explicit CSeedDropoutDesc(IMathEngine& mathEngine, bool isMask);

	static constexpr int cacheSize = 64;
	static constexpr int maskAlign = 4;
	static constexpr int numOfGenerations = (cacheSize + (maskAlign - 1)) / maskAlign;
};

} // namespace NeoML
