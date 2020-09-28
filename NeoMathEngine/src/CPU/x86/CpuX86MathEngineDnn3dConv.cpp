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

#include <common.h>
#pragma hdrstop

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_SSE

#include <CpuMathEngine.h>
#include <CpuX86.h>
#include <float.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnConv.h>
#include <CpuX86MathEngineBlasPrivate.h>
#include <CpuX86MathEngineVectorMathPrivate.h>

namespace NeoML {

void CCpuMathEngine::blob3dConvolution1x1x1(  const CBlobDesc& source, const CBlobDesc& filter, const CBlobDesc& result,
	int strideHeight, int strideWidth, int strideDepth,
	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData )
{
	static constexpr int goodDenominatorFirst = 2;
	static constexpr int goodDenominatorSecond = 2;
	const int channels = source.Channels();
	const int geomSize = result.ObjectCount() * result.GeometricalSize();
	const int newChannels = result.Channels();
	// Convolution is matrix product
	// [geomSize x channels] * [newChannels x channels]T
	// then add the free term if necessary

	const auto opCount = static_cast<int64_t>(source.BlobSize()) * static_cast<int64_t>(filter.BlobSize());
	if( strideHeight == 1 && strideWidth == 1 && strideDepth == 1) {
		if( geomSize > newChannels ) {
			// The first matrix split into rows
			NEOML_OMP_NUM_THREADS(IsOmpRelevant(geomSize, opCount) ? threadCount : 1)
			{
				int geomStart;
				int geomCount;
				if( OmpGetTaskIndexAndCount(geomSize, goodDenominatorFirst, geomStart, geomCount) ) {
					float* outputDataPtr = resultData + geomStart * newChannels;
					if( freeTermData != 0 ) {
						NeoML::setVectorToMatrixRows(outputDataPtr, geomCount, newChannels, freeTermData);
					} else {
						NeoML::vectorFill(outputDataPtr, 0, geomCount * newChannels);
					}
					multiplyMatrixByTransposedMatrixAndAdd(sourceData + geomStart * channels,
						geomCount, channels, channels,
						filterData, newChannels, channels,
						outputDataPtr, newChannels);
				}
			}
		} else {
			// The second matrix split into rows
			NEOML_OMP_NUM_THREADS(IsOmpRelevant(newChannels, opCount) ? threadCount : 1)
			{
				int channelStart;
				int channelCount;
				if( OmpGetTaskIndexAndCount(newChannels, goodDenominatorSecond, channelStart, channelCount) ) {
					float* resultPtr = resultData + channelStart;
					float* resultEnd = resultPtr + newChannels * geomSize;
					if( freeTermData != 0 ) {
						const float* freeTerm = freeTermData + channelStart;
						for( float* res = resultPtr; res < resultEnd; res += newChannels ) {
							dataCopy(res, freeTerm, channelCount);
						}
					} else {
						for( float* res = resultPtr; res < resultEnd; res += newChannels ) {
							NeoML::vectorFill(res, 0, channelCount);
						}
					}
					multiplyMatrixByTransposedMatrixAndAdd(sourceData,
						geomSize, channels, channels,
						filterData + channelStart * channels, channelCount, channels,
						resultData + channelStart, newChannels);
				}
			}
		}
	} else {
		CFloatHandleVar repackedHolder(mathEngine(), geomSize * channels);
		float* repackedData = GetRaw(repackedHolder.GetHandle());

		NEOML_OMP_NUM_THREADS(IsOmpRelevant(geomSize, opCount) ? threadCount : 1)
	{
			int geomStart;
			int geomCount;
			if( OmpGetTaskIndexAndCount(geomSize, geomStart, geomCount) ) {
				// Repack the input blob, removing the unused data
				for( int out = geomStart; out < geomStart + geomCount; ++out ) {
					int objNum = out;
					int outK = objNum % result.Depth();
					objNum /= result.Depth();
					int outI = objNum % result.Width();
					objNum /= result.Width();
					int outJ = objNum % result.Height();
					objNum /= result.Height();

					float* sourceDataPtr = repackedData + out * channels;
					const float* inputData = sourceData + ( ( ( objNum * source.Height() + outJ * strideHeight )
						* source.Width() + outI * strideWidth) * source.Depth() + outK * strideDepth) * channels;
					dataCopy(sourceDataPtr, inputData, channels);
				}
			
				float* outputDataPtr = resultData + geomStart * newChannels;
			if( freeTermData != 0 ) {
					NeoML::setVectorToMatrixRows(outputDataPtr, geomCount, newChannels, freeTermData);
			} else {
					NeoML::vectorFill(outputDataPtr, 0, geomCount * newChannels);
			}
				multiplyMatrixByTransposedMatrixAndAdd(repackedData + geomStart * channels,
					geomCount, channels, channels,
					filterData, newChannels, channels,
					outputDataPtr, newChannels);
		}
	}
}
}

} // namespace NeoML

#endif