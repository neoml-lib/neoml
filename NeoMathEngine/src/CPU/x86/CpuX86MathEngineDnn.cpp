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
#include <MathEngineDnnPoolings.h>

namespace NeoML {

void CCpuMathEngine::BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle& maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonGlobalMaxPoolingDesc& desc = static_cast<const CCommonGlobalMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;
	const CBlobDesc& maxIndices = desc.MaxIndices;

	int poolSize = source.Height() * source.Width() * source.Depth();
	int maxCount = result.Height() * result.Width() * result.Depth();

	const float* sourcePtr = GetRaw( sourceData );
	int* maxIndexPtr = GetRaw( maxIndicesData );
	float* resultPtr = GetRaw( resultData );
	int resultObjectSize = maxCount * result.Channels();

	VectorFill(maxIndicesData, -1, result.BlobSize());
	VectorFill(resultData, -FLT_MAX, result.BlobSize());

	int sseChannels;
	int nonSseChannels;
	checkSse2(source.Channels(), sseChannels, nonSseChannels);

	for(int b = 0; b < source.ObjectCount(); ++b) {
		for(int i = 0; i < poolSize; ++i) {
			int* maxIndexItem = maxIndexPtr;
			float* resultItem = resultPtr;

			if(sseChannels > 0) {
				__m128i iSse = _mm_set1_epi32(i);
				for(int c = 0; c < sseChannels; ++c) {
					__m128 nextVal = _mm_loadu_ps(sourcePtr);
					__m128i nextIndex = iSse;

					int* maxIndexCheck = maxIndexItem;
					float* resultCheck = resultItem;
					for(int check = 0; check < maxCount; ++check) {
						__m128 curVal = _mm_loadu_ps(resultCheck);
						__m128 compare = _mm_cmpge_ps(nextVal, curVal);

						if(_mm_movemask_ps(compare) != 0) {
							__m128i iCompare = _mm_castps_si128(compare);
							__m128i curIndex = _mm_loadu_si128((const __m128i*)maxIndexCheck);

							__m128 curValSet = _mm_or_ps(_mm_andnot_ps(compare, curVal), _mm_and_ps(compare, nextVal));
							_mm_storeu_ps(resultCheck, curValSet);
							__m128i curIndexSet = _mm_or_si128(_mm_andnot_si128(iCompare, curIndex),
								_mm_and_si128(iCompare, nextIndex));
							_mm_storeu_si128((__m128i*)maxIndexCheck, curIndexSet);

							if(check < maxCount - 1) {
								nextVal = _mm_or_ps(_mm_andnot_ps(compare, nextVal), _mm_and_ps(compare, curVal));
								nextIndex = _mm_or_si128(_mm_andnot_si128(iCompare, nextIndex),
									_mm_and_si128(iCompare, curIndex));
							}
						}

						maxIndexCheck += maxIndices.Channels();
						resultCheck += result.Channels();
					}

					maxIndexItem += 4;
					resultItem += 4;
					sourcePtr += 4;
				}
			}

			for(int c = 0; c < nonSseChannels; ++c) {
				float value = *sourcePtr++;

				int* maxIndexCheck = maxIndexItem;
				float* resultCheck = resultItem;
				for(int check = 0; check < maxCount; ++check) {
					if(value >= *resultCheck) {
						float nextVal = value;
						int nextIndex = i;
						for(int set = check; set < maxCount; ++set) {
							float preVal = *resultCheck;
							int preIndex = *maxIndexCheck;
							*resultCheck = nextVal;
							*maxIndexCheck = nextIndex;
							nextVal = preVal;
							nextIndex = preIndex;

							maxIndexCheck += maxIndices.Channels();
							resultCheck += result.Channels();
						}
						break;
					}

					maxIndexCheck += maxIndices.Channels();
					resultCheck += result.Channels();
				}

				++maxIndexItem;
				++resultItem;
			}
		}

		maxIndexPtr += resultObjectSize;
		resultPtr += resultObjectSize;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// 3d max pooling

static void blob3dMaxPoolingProcessFirstItem(const float* sourceObject, int sourceIndex,
	int sseChannels, int nonSseChannels, float* resultData, int* indexData)
{
	const float* sourceData = sourceObject + sourceIndex;

	if(sseChannels > 0) {
		__m128i sourceIndexSse = _mm_set1_epi32(sourceIndex);
		for(int c = 0; c < sseChannels; ++c) {
			_mm_storeu_ps(resultData, _mm_loadu_ps(sourceData));
			_mm_storeu_si128((__m128i*)indexData, sourceIndexSse);

			sourceData += 4;
			resultData += 4;
			indexData += 4;
		}
	}

	for(int c = 0; c < nonSseChannels; ++c) {
		*resultData++ = *sourceData++;
		*indexData++ = sourceIndex;
	}
}

static void blob3dMeanMaxPoolingProcessFirstItem(const float* sourceObject, int sourceIndex,
	int sseChannels, int nonSseChannels, float* resultData)
{
	const float* sourceData = sourceObject + sourceIndex;

	for(int c = 0; c < sseChannels; ++c) {
		_mm_storeu_ps(resultData, _mm_loadu_ps(sourceData));

		sourceData += 4;
		resultData += 4;
	}

	for(int c = 0; c < nonSseChannels; ++c) {
		*resultData++ = *sourceData++;
	}
}

static void blob3dMaxPoolingProcessItem(const float* sourceObject, int sourceIndex,
	int sseChannels, int nonSseChannels, float* resultData, int* indexData)
{
	const float* sourceData = sourceObject + sourceIndex;

	if(sseChannels > 0) {
		__m128i sourceIndexSse = _mm_set1_epi32(sourceIndex);
		for(int c = 0; c < sseChannels; ++c) {
			__m128 res = _mm_loadu_ps(resultData);
			__m128 src = _mm_loadu_ps(sourceData);
			__m128i index = _mm_loadu_si128((const __m128i*)indexData);

			__m128 cmp = _mm_cmplt_ps(res, src);

			res = _mm_or_ps(_mm_andnot_ps(cmp, res), _mm_and_ps(cmp, src));
			_mm_storeu_ps(resultData, res);

			__m128i iCmp = _mm_castps_si128(cmp);
			index = _mm_or_si128(_mm_andnot_si128(iCmp, index), _mm_and_si128(iCmp, sourceIndexSse));
			_mm_storeu_si128((__m128i*)indexData, index);

			sourceData += 4;
			resultData += 4;
			indexData += 4;
		}
	}

	for(int c = 0; c < nonSseChannels; ++c) {
		if(*resultData < *sourceData) {
			*resultData = *sourceData;
			*indexData = sourceIndex;
		}
		++sourceData;
		++resultData;
		++indexData;
	}
}

static void blob3dMaxPoolingProcessItem(const float* sourceObject, int sourceIndex,
	int sseChannels, int nonSseChannels, float* resultData)
{
	const float* sourceData = sourceObject + sourceIndex;

	for(int c = 0; c < sseChannels; ++c) {
		__m128 res = _mm_loadu_ps(resultData);
		__m128 src = _mm_loadu_ps(sourceData);

		res = _mm_max_ps(res, src);
		_mm_storeu_ps(resultData, res);

		sourceData += 4;
		resultData += 4;
	}

	for(int c = 0; c < nonSseChannels; ++c) {
		if(*resultData < *sourceData) {
			*resultData = *sourceData;
		}
		++sourceData;
		++resultData;
	}
}

void CCpuMathEngine::Blob3dMaxPooling( const C3dMaxPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommon3dMaxPoolingDesc& desc = static_cast<const CCommon3dMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const float* sourceObject = GetRaw( sourceData );
	float* resultJStart = GetRaw( resultData );
	int* indexJStart = (maxIndicesData == 0) ? 0 : GetRaw( *maxIndicesData );

	int sourceDepthSize = source.Depth() * source.Channels();
	int sourceRowSize = source.Width() * sourceDepthSize;
	int sourceObjectSize = source.Height() * sourceRowSize;

	int resultDepthSize = result.Depth() * result.Channels();
	int resultRowSize = result.Width() * resultDepthSize;

	int sseChannels;
	int nonSseChannels;
	checkSse2(result.Channels(), sseChannels, nonSseChannels);

	for(int b = 0; b < result.ObjectCount(); ++b) {
		// Go through all "cube blocks" and then go over values in each block
		// So we get a forward pass through the input and filterHeight * filterWidth passes through the output
		for(int j = 0; j < result.Height(); ++j) {
			int sourceJIndex = j * desc.StrideHeight * sourceRowSize;
			for(int filterJ = 0; filterJ < desc.FilterHeight; ++filterJ) {
				float* resultIStart = resultJStart;
				int* indexIStart = indexJStart;

				for(int i = 0; i < result.Width(); ++i) {
					int sourceIIndex = sourceJIndex + i * desc.StrideWidth * sourceDepthSize;
					for(int filterI = 0; filterI < desc.FilterWidth; ++filterI) {
						float* resultDataPtr = resultIStart;
						int* indexData = indexIStart;

						for(int k = 0; k < result.Depth(); ++k) {
							int sourceIndex = sourceIIndex + k * desc.StrideDepth * source.Channels();
							for(int filterK = 0; filterK < desc.FilterDepth; ++filterK) {
								if((filterJ == 0) && (filterI == 0) && (filterK == 0)) {
									if(indexData == 0) {
										blob3dMeanMaxPoolingProcessFirstItem(sourceObject, sourceIndex,
											sseChannels, nonSseChannels, resultDataPtr);
									} else {
										blob3dMaxPoolingProcessFirstItem(sourceObject, sourceIndex,
											sseChannels, nonSseChannels, resultDataPtr, indexData);
									}
								} else {
									if(indexData == 0) {
										blob3dMaxPoolingProcessItem(sourceObject, sourceIndex,
											sseChannels, nonSseChannels, resultDataPtr);
									} else {
										blob3dMaxPoolingProcessItem(sourceObject, sourceIndex,
											sseChannels, nonSseChannels, resultDataPtr, indexData);
									}
								}

								sourceIndex += source.Channels();
							}
							resultDataPtr += result.Channels();
							if(indexData != 0) {
								indexData += result.Channels();
							}
						}
						sourceIIndex += sourceDepthSize;
					}
					resultIStart += resultDepthSize;
					if(indexIStart != 0) {
						indexIStart += resultDepthSize;
					}
				}
				sourceJIndex += sourceRowSize;
			}
			resultJStart += resultRowSize;
			if(indexJStart != 0) {
				indexJStart += resultRowSize;
			}
		}

		sourceObject += sourceObjectSize;
	}
}

//////////////////////////////////
// 3d mean pooling

static void blob3dMeanPoolingProcessItem(const float* sourceObject, int sourceIndex,
	int sseChannels, int nonSseChannels, float* resultData)
{
	const float* sourceData = sourceObject + sourceIndex;

	for(int c = 0; c < sseChannels; ++c) {
		__m128 src = _mm_loadu_ps(sourceData);
		__m128 res = _mm_loadu_ps(resultData);
		_mm_storeu_ps(resultData, _mm_add_ps(res, src));

		sourceData += 4;
		resultData += 4;
	}

	for(int c = 0; c < nonSseChannels; ++c) {
		*resultData++ += *sourceData++;
	}
}

void CCpuMathEngine::Blob3dMeanPooling( const C3dMeanPoolingDesc& convDesc, const CFloatHandle& sourceData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommon3dMeanPoolingDesc& desc = static_cast<const CCommon3dMeanPoolingDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const float* sourceObject = GetRaw( sourceData );
	float* resultJStart = GetRaw( resultData );

	int sourceDepthSize = source.Depth() * source.Channels();
	int sourceRowSize = source.Width() * sourceDepthSize;
	int sourceObjectSize = source.Height() * sourceRowSize;

	int resultDepthSize = result.Depth() * result.Channels();
	int resultRowSize = result.Width() * resultDepthSize;

	int sseChannels;
	int nonSseChannels;
	checkSse(result.Channels(), sseChannels, nonSseChannels);

	for(int b = 0; b < result.ObjectCount(); ++b) {
		// Go through all "cube blocks" and then go over values in each block
		// So we get a forward pass through the input and filterHeight * filterWidth passes through the output
		for(int j = 0; j < result.Height(); ++j) {
			int sourceJIndex = j * desc.StrideHeight * sourceRowSize;
			for(int filterJ = 0; filterJ < desc.FilterHeight; ++filterJ) {
				float* resultIStart = resultJStart;

				for(int i = 0; i < result.Width(); ++i) {
					int sourceIIndex = sourceJIndex + i * desc.StrideWidth * sourceDepthSize;
					for(int filterI = 0; filterI < desc.FilterWidth; ++filterI) {
						float* resultDataPtr = resultIStart;

						for(int k = 0; k < result.Depth(); ++k) {
							int sourceIndex = sourceIIndex + k * desc.StrideDepth * source.Channels();
							for(int filterK = 0; filterK < desc.FilterDepth; ++filterK) {
								if((filterJ == 0) && (filterI == 0) && (filterK == 0)) {
									blob3dMeanMaxPoolingProcessFirstItem(sourceObject, sourceIndex,
										sseChannels, nonSseChannels, resultDataPtr);
								} else {
									blob3dMeanPoolingProcessItem(sourceObject, sourceIndex,
										sseChannels, nonSseChannels, resultDataPtr);
								}

								sourceIndex += source.Channels();
							}
							resultDataPtr += result.Channels();
						}
						sourceIIndex += sourceDepthSize;
					}
					resultIStart += resultDepthSize;
				}
				sourceJIndex += sourceRowSize;
			}
			resultJStart += resultRowSize;
		}

		sourceObject += sourceObjectSize;
	}

	// Divide the output by filter volume
	CFloatHandleStackVar denom( mathEngine(), 1 );
	denom.SetValue( 1.f / desc.FilterHeight / desc.FilterWidth / desc.FilterDepth );
	VectorMultiply( resultData, resultData, result.BlobSize(), denom );
}

static void applyMeanPoolingBackwardSet(const float* outputDiffData, int sseChannels, int nonSseChannels,
	int filterHeight, int filterWidth, int filterDepth, float* inputDiffData,
	int inputRowSize, int inputDepthSize)
{
	for(int j = 0; j < filterHeight; ++j) {
		float* inputDiffDepth = inputDiffData;
		for(int i = 0; i < filterWidth; ++i) {
			float* inputDiffPixel = inputDiffDepth;
			for(int k = 0; k < filterDepth; ++k) {
				const float* outputDiffPixel = outputDiffData;
				for(int c = 0; c < sseChannels; ++c) {
					_mm_storeu_ps(inputDiffPixel, _mm_loadu_ps(outputDiffPixel));
					inputDiffPixel += 4;
					outputDiffPixel += 4;
				}
				for(int c = 0; c < nonSseChannels; ++c) {
					*inputDiffPixel++ = *outputDiffPixel++;
				}
			}
			inputDiffDepth += inputDepthSize;
		}
		inputDiffData += inputRowSize;
	}
}

static void applyMeanPoolingBackwardAdd(const float* outputDiffData, int sseChannels, int nonSseChannels,
	int filterHeight, int filterWidth, int filterDepth, float* inputDiffData,
	int inputRowSize, int inputDepthSize)
{
	for(int j = 0; j < filterHeight; ++j) {
		float* inputDiffDepth = inputDiffData;
		for(int i = 0; i < filterWidth; ++i) {
			float* inputDiffPixel = inputDiffDepth;
			for(int k = 0; k < filterDepth; ++k) {
				const float* outputDiffPixel = outputDiffData;
				for(int c = 0; c < sseChannels; ++c) {
					__m128 src = _mm_loadu_ps(outputDiffPixel);
					__m128 res = _mm_loadu_ps(inputDiffPixel);
					_mm_storeu_ps(inputDiffPixel, _mm_add_ps(res, src));
					inputDiffPixel += 4;
					outputDiffPixel += 4;
				}
				for(int c = 0; c < nonSseChannels; ++c) {
					*inputDiffPixel++ += *outputDiffPixel++;
				}
			}
			inputDiffDepth += inputDepthSize;
		}
		inputDiffData += inputRowSize;
	}
}

void CCpuMathEngine::Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc& poolingDesc,
	const CFloatHandle& outputDiffData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCommon3dMeanPoolingDesc& desc = static_cast<const CCommon3dMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;

	if( desc.FilterHeight != desc.StrideHeight || desc.FilterWidth != desc.StrideWidth || desc.FilterDepth != desc.StrideDepth ) {
		// Either the cube blocks for pooling have nonzero intersections and several diffs should be added up
		// or some of the data is skipped when pooling, and diff should be set to 0 for them
		VectorFill( inputDiffData, 0, inputDiff.BlobSize() );
	}

	// The flag that indicates that the cube blocks for pooling have nonzero intersections
	bool isIntersect = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth || desc.FilterDepth > desc.StrideDepth;

	int sseChannels;
	int nonSseChannels;
	checkSse(outputDiff.Channels(), sseChannels, nonSseChannels);

	int inputDepthSize = inputDiff.Depth() * inputDiff.Channels();
	int inputRowSize = inputDepthSize * inputDiff.Width();
	int inputObjectSize = inputRowSize * inputDiff.Height();

	const float* outputDiffDataPtr = GetRaw( outputDiffData );
	float* inputDiffDataPtr = GetRaw( inputDiffData );

	for(int b = 0; b < outputDiff.ObjectCount(); ++b) {
		int jStart = 0;
		for(int j = 0; j < outputDiff.Height(); ++j) {
			int iStart = jStart;
			for(int i = 0; i < outputDiff.Width(); ++i) {
				int kStart = iStart;
				for(int k = 0; k < outputDiff.Depth(); ++k) {
					if(isIntersect) {
						applyMeanPoolingBackwardAdd(outputDiffDataPtr, sseChannels, nonSseChannels,
							desc.FilterHeight, desc.FilterWidth, desc.FilterDepth, inputDiffDataPtr + kStart,
							inputRowSize, inputDepthSize);
					} else {
						applyMeanPoolingBackwardSet(outputDiffDataPtr, sseChannels, nonSseChannels,
							desc.FilterHeight, desc.FilterWidth, desc.FilterDepth, inputDiffDataPtr + kStart,
							inputRowSize, inputDepthSize);
					}
					outputDiffDataPtr += outputDiff.Channels();
					kStart += inputDiff.Channels() * desc.StrideDepth;
				}
				iStart += inputDepthSize * desc.StrideWidth;
			}
			jStart += inputRowSize * desc.StrideHeight;
		}
		inputDiffDataPtr += inputObjectSize;
	}

	// Divide the output by the filter volume
	CFloatHandleStackVar denom( mathEngine(), 1 );
	denom.SetValue( 1.f / desc.FilterHeight / desc.FilterWidth / desc.FilterDepth );
	VectorMultiply( inputDiffData, inputDiffData, inputDiff.BlobSize(), denom );
}

void CCpuMathEngine::BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonMaxOverTimePoolingDesc& desc = static_cast<const CCommonMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	if( maxIndicesData != 0 ) {
		int seqElemSize = source.BlobSize() / source.BatchLength();
		int seqElemSizeSse = seqElemSize / 4;
		int seqElemSizeNonSse = seqElemSize % 4;
	
		CConstFloatHandle sourceStart = sourceData;
		CFloatHandle resultStart = resultData;
		CIntHandle indexStart = *maxIndicesData;

		int indexValueStart = 0;
		for(int l = 0; l < result.BatchLength(); ++l) {
			const float* sourceDataPtr = GetRaw( sourceStart );

			// Set the initial values (the zero value)
			VectorCopy(resultStart, sourceStart, seqElemSize);
			VectorFill(indexStart, indexValueStart, seqElemSize);

			sourceDataPtr += seqElemSize;

			for(int n = 1; n < desc.FilterLen; ++n) {
				// Restart the result data
				float* resultDataPtr = GetRaw( resultStart );
				int* indexData = GetRaw( indexStart );
				int indexValue = indexValueStart + n;
				__m128i indexValueSse = _mm_set1_epi32(indexValue);

				for(int i = 0; i < seqElemSizeSse; ++i) {
					__m128 src = _mm_loadu_ps(sourceDataPtr);
					__m128 res = _mm_loadu_ps(resultDataPtr);
					__m128i cmp = _mm_castps_si128(_mm_cmpgt_ps(src, res));

					res = _mm_max_ps(res, src);
					_mm_storeu_ps(resultDataPtr, res);

					__m128i ind = _mm_loadu_si128((const __m128i*)indexData);
					ind = _mm_or_si128(_mm_andnot_si128(cmp, ind), _mm_and_si128(cmp, indexValueSse));
					_mm_storeu_si128((__m128i*)indexData, ind);

					sourceDataPtr += 4;
					resultDataPtr += 4;
					indexData += 4;
				}
				for(int i = 0; i < seqElemSizeNonSse; ++i) {
					if(*sourceDataPtr > *resultDataPtr) {
						*resultDataPtr = *sourceDataPtr;
						*indexData = indexValue;
					}
					++sourceDataPtr;
					++resultDataPtr;
					++indexData;
				}
			}
			sourceStart += desc.StrideLen * seqElemSize;
			resultStart += seqElemSize;
			indexStart += seqElemSize;

			indexValueStart += desc.StrideLen;
		}
	} else {
		int seqElemSize = source.ObjectSize() * source.BatchWidth();
		int seqElemSizeSse = seqElemSize / 4;
		int seqElemSizeNonSse = seqElemSize % 4;

		CConstFloatHandle sourceStart = sourceData;
		CFloatHandle resultStart = resultData;

		for(int l = 0; l < result.BatchLength(); ++l) {
			const float* sourceDataPtr = GetRaw( sourceStart );

			// Set the initial values (the zero value)
			VectorCopy( resultStart, sourceStart, seqElemSize);

			sourceDataPtr += seqElemSize;

			for(int n = 1; n < desc.FilterLen; ++n) {
				// Restart the result data
				float* resultDataPtr = GetRaw(resultStart);

				for(int i = 0; i < seqElemSizeSse; ++i) {
					__m128 src = _mm_loadu_ps(sourceDataPtr);
					__m128 res = _mm_loadu_ps(resultDataPtr);

					res = _mm_max_ps(res, src);
					_mm_storeu_ps(resultDataPtr, res);

					sourceDataPtr += 4;
					resultDataPtr += 4;
				}

				for(int i = 0; i < seqElemSizeNonSse; ++i) {
					if(*sourceDataPtr > *resultDataPtr) {
						*resultDataPtr = *sourceDataPtr;
					}
					++sourceDataPtr;
					++resultDataPtr;
				}
			}

			sourceStart += desc.StrideLen * seqElemSize;
			resultStart += seqElemSize;
		}
	}
}

void CCpuMathEngine::AddWidthIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const float* pSource = GetRaw( sourceData );
	float* pResult = GetRaw( resultData );

	for( int batch = 0; batch < source.ObjectCount(); ++batch ) {
		for( int h = 0; h < source.Height(); ++h ) {
			for( int w = 0; w < source.Width(); ++w ) {
				for( int c = 0; c < source.Channels(); c++ ) {
					*pResult = isForward ? *pSource + w : *pSource - w;
					pSource++;
					pResult++;
				}
			}
		}
	}
}

void CCpuMathEngine::AddWidthIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const int* pSource = GetRaw( sourceData );
	int* pResult = GetRaw( resultData );

	for( int batch = 0; batch < source.ObjectCount(); ++batch ) {
		for( int h = 0; h < source.Height(); ++h ) {
			for( int w = 0; w < source.Width(); ++w ) {
				for( int c = 0; c < source.Channels(); c++ ) {
					*pResult = isForward ? *pSource + w : *pSource - w;
					pSource++;
					pResult++;
				}
			}
		}
	}
}

void CCpuMathEngine::AddHeightIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const float* pSource = GetRaw( sourceData );
	float* pResult = GetRaw( resultData );

	for( int batch = 0; batch < source.ObjectCount(); ++batch ) {
		for( int h = 0; h < source.Height(); ++h ) {
			for( int w = 0; w < source.Width(); ++w ) {
				for( int c = 0; c < source.Channels(); c++ ) {
					*pResult = isForward ? *pSource + h : *pSource - h;
					pSource++;
					pResult++;
				}
			}
		}
	}
}

void CCpuMathEngine::AddHeightIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const int* pSource = GetRaw( sourceData );
	int* pResult = GetRaw( resultData );

	for( int batch = 0; batch < source.ObjectCount(); ++batch ) {
		for( int h = 0; h < source.Height(); ++h ) {
			for( int w = 0; w < source.Width(); ++w ) {
				for( int c = 0; c < source.Channels(); c++ ) {
					*pResult = isForward ? *pSource + h : *pSource - h;
					pSource++;
					pResult++;
				}
			}
		}
	}
}

} // namespace NeoML

#endif // NEOML_USE_SSE
