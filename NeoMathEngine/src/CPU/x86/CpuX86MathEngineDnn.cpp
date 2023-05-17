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

#include <common.h>
#pragma hdrstop

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_SSE

#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <CpuMathEnginePrivate.h>
#include <float.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnPoolings.h>

namespace NeoML {

void CCpuMathEngine::BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CIntHandle& maxIndices, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonGlobalMaxPoolingDesc& desc = static_cast<const CCommonGlobalMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;
	const CBlobDesc& maxIndex = desc.MaxIndices;

	const int poolSize = source.Height() * source.Width() * source.Depth();
	const int maxCount = result.Height() * result.Width() * result.Depth();

	const float* sourcePtr = GetRaw( sourceData );
	int* maxIndexPtr = GetRaw( maxIndices );
	float* resultPtr = GetRaw( resultData );
	const int resultObjectSize = maxCount * result.Channels();

	VectorFill( maxIndices, -1, result.BlobSize() );
	VectorFill( resultData, -FLT_MAX, result.BlobSize() );

	int sseChannels;
	int nonSseChannels;
	checkSse2( source.Channels(), sseChannels, nonSseChannels );

	for( int b = 0; b < source.ObjectCount(); ++b ) {
		for( int i = 0; i < poolSize; ++i ) {
			int* maxIndexItem = maxIndexPtr;
			float* resultItem = resultPtr;

			if( sseChannels > 0 ) {
				__m128i iSse = _mm_set1_epi32( i );
				for( int c = 0; c < sseChannels; ++c ) {
					__m128 nextVal = _mm_loadu_ps( sourcePtr );
					__m128i nextIndex = iSse;

					int* maxIndexCheck = maxIndexItem;
					float* resultCheck = resultItem;
					for( int check = 0; check < maxCount; ++check ) {
						__m128 curVal = _mm_loadu_ps( resultCheck );
						__m128 compare = _mm_cmpge_ps( nextVal, curVal );

						if( _mm_movemask_ps( compare ) != 0 ) {
							__m128i iCompare = _mm_castps_si128( compare );
							__m128i curIndex = _mm_loadu_si128( ( const __m128i* )maxIndexCheck );

							__m128 curValSet = _mm_or_ps( _mm_andnot_ps( compare, curVal ), _mm_and_ps( compare, nextVal ) );
							_mm_storeu_ps( resultCheck, curValSet );
							__m128i curIndexSet = _mm_or_si128( _mm_andnot_si128( iCompare, curIndex ),
								_mm_and_si128( iCompare, nextIndex ) );
							_mm_storeu_si128( ( __m128i* )maxIndexCheck, curIndexSet );

							if( check < maxCount - 1 ) {
								nextVal = _mm_or_ps( _mm_andnot_ps( compare, nextVal ), _mm_and_ps( compare, curVal ) );
								nextIndex = _mm_or_si128( _mm_andnot_si128( iCompare, nextIndex ),
									_mm_and_si128( iCompare, curIndex ) );
							}
						}

						maxIndexCheck += maxIndex.Channels();
						resultCheck += result.Channels();
					}

					maxIndexItem += 4;
					resultItem += 4;
					sourcePtr += 4;
				}
			}

			for( int c = 0; c < nonSseChannels; ++c ) {
				float value = *sourcePtr++;

				int* maxIndexCheck = maxIndexItem;
				float* resultCheck = resultItem;
				for( int check = 0; check < maxCount; ++check ) {
					if( value >= *resultCheck ) {
						float nextVal = value;
						int nextIndex = i;
						for( int set = check; set < maxCount; ++set ) {
							float preVal = *resultCheck;
							int preIndex = *maxIndexCheck;
							*resultCheck = nextVal;
							*maxIndexCheck = nextIndex;
							nextVal = preVal;
							nextIndex = preIndex;

							maxIndexCheck += maxIndex.Channels();
							resultCheck += result.Channels();
						}
						break;
					}
					maxIndexCheck += maxIndex.Channels();
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

//-------------------------------------------------------------------------------------------------------
// 3d max pooling

static void blob3dMaxPoolingProcessFirstItem( const float* sourceObject, int sourceIndex,
	int sseChannels, int nonSseChannels, float* resultData, int* indexData )
{
	const float* sourceData = sourceObject + sourceIndex;

	if( sseChannels > 0 ) {
		__m128i sourceIndexSse = _mm_set1_epi32( sourceIndex );
		for( int c = 0; c < sseChannels; ++c ) {
			_mm_storeu_ps( resultData, _mm_loadu_ps( sourceData ) );
			_mm_storeu_si128( ( __m128i* )indexData, sourceIndexSse );

			sourceData += 4;
			resultData += 4;
			indexData += 4;
		}
	}

	for( int c = 0; c < nonSseChannels; ++c ) {
		*resultData++ = *sourceData++;
		*indexData++ = sourceIndex;
	}
}

static void blob3dMeanMaxPoolingProcessFirstItem( const float* sourceObject, int sourceIndex,
	int sseChannels, int nonSseChannels, float* resultData )
{
	const float* sourceData = sourceObject + sourceIndex;

	for( int c = 0; c < sseChannels; ++c ) {
		_mm_storeu_ps( resultData, _mm_loadu_ps( sourceData ) );

		sourceData += 4;
		resultData += 4;
	}

	for( int c = 0; c < nonSseChannels; ++c ) {
		*resultData++ = *sourceData++;
	}
}

static void blob3dMaxPoolingProcessItem( const float* sourceObject, int sourceIndex,
	int sseChannels, int nonSseChannels, float* resultData, int* indexData )
{
	const float* sourceData = sourceObject + sourceIndex;

	if( sseChannels > 0 ) {
		__m128i sourceIndexSse = _mm_set1_epi32( sourceIndex );
		for( int c = 0; c < sseChannels; ++c ) {
			__m128 res = _mm_loadu_ps( resultData );
			__m128 src = _mm_loadu_ps( sourceData );
			__m128i index = _mm_loadu_si128( ( const __m128i* )indexData );

			__m128 cmp = _mm_cmplt_ps( res, src );

			res = _mm_or_ps( _mm_andnot_ps( cmp, res ), _mm_and_ps( cmp, src ) );
			_mm_storeu_ps( resultData, res );

			__m128i iCmp = _mm_castps_si128( cmp );
			index = _mm_or_si128( _mm_andnot_si128( iCmp, index ), _mm_and_si128( iCmp, sourceIndexSse ) );
			_mm_storeu_si128( ( __m128i* )indexData, index );

			sourceData += 4;
			resultData += 4;
			indexData += 4;
		}
	}

	for( int c = 0; c < nonSseChannels; ++c ) {
		if( *resultData < *sourceData ) {
			*resultData = *sourceData;
			*indexData = sourceIndex;
		}
		++sourceData;
		++resultData;
		++indexData;
	}
}

static void blob3dMaxPoolingProcessItem( const float* sourceObject, int sourceIndex,
	int sseChannels, int nonSseChannels, float* resultData )
{
	const float* sourceData = sourceObject + sourceIndex;

	for( int c = 0; c < sseChannels; ++c ) {
		__m128 res = _mm_loadu_ps( resultData );
		__m128 src = _mm_loadu_ps( sourceData );

		res = _mm_max_ps( res, src );
		_mm_storeu_ps( resultData, res );

		sourceData += 4;
		resultData += 4;
	}

	for( int c = 0; c < nonSseChannels; ++c ) {
		if( *resultData < *sourceData ) {
			*resultData = *sourceData;
		}
		++sourceData;
		++resultData;
	}
}

void CCpuMathEngine::Blob3dMaxPooling( const C3dMaxPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CIntHandle* maxIndices, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices == 0 || maxIndices->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommon3dMaxPoolingDesc& desc = static_cast<const CCommon3dMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const float* sourceObject = GetRaw( sourceData );
	float* resultJStart = GetRaw( resultData );
	int* indexJStart = ( maxIndices == 0 ) ? 0 : GetRaw( *maxIndices );

	const int sourceDepthSize = source.Depth() * source.Channels();
	const int sourceRowSize = source.Width() * sourceDepthSize;
	const int sourceObjectSize = source.Height() * sourceRowSize;

	const int resultDepthSize = result.Depth() * result.Channels();
	const int resultRowSize = result.Width() * resultDepthSize;

	int sseChannels;
	int nonSseChannels;
	checkSse2( result.Channels(), sseChannels, nonSseChannels );

	for( int b = 0; b < result.ObjectCount(); ++b ) {
		// Go through all "cube blocks" and then go over values in each block
		// So we get a forward pass through the source and filterHeight * filterWidth passes through the result
		for( int j = 0; j < result.Height(); ++j ) {
			int sourceJIndex = j * desc.StrideHeight * sourceRowSize;
			for( int filterJ = 0; filterJ < desc.FilterHeight; ++filterJ ) {
				float* resultIStart = resultJStart;
				int* indexIStart = indexJStart;

				for( int i = 0; i < result.Width(); ++i ) {
					int sourceIIndex = sourceJIndex + i * desc.StrideWidth * sourceDepthSize;
					for( int filterI = 0; filterI < desc.FilterWidth; ++filterI ) {
						float* resultDataPtr = resultIStart;
						int* indexData = indexIStart;

						for( int k = 0; k < result.Depth(); ++k ) {
							int sourceIndex = sourceIIndex + k * desc.StrideDepth * source.Channels();
							for( int filterK = 0; filterK < desc.FilterDepth; ++filterK ) {
								if( ( filterJ == 0 ) && ( filterI == 0 ) && ( filterK == 0 ) ) {
									if( indexData == 0 ) {
										blob3dMeanMaxPoolingProcessFirstItem( sourceObject, sourceIndex,
											sseChannels, nonSseChannels, resultDataPtr );
									} else {
										blob3dMaxPoolingProcessFirstItem( sourceObject, sourceIndex,
											sseChannels, nonSseChannels, resultDataPtr, indexData );
									}
								} else {
									if( indexData == 0 ) {
										blob3dMaxPoolingProcessItem( sourceObject, sourceIndex,
											sseChannels, nonSseChannels, resultDataPtr );
									} else {
										blob3dMaxPoolingProcessItem( sourceObject, sourceIndex,
											sseChannels, nonSseChannels, resultDataPtr, indexData );
									}
								}

								sourceIndex += source.Channels();
							}
							resultDataPtr += result.Channels();
							if( indexData != 0 ) {
								indexData += result.Channels();
							}
						}
						sourceIIndex += sourceDepthSize;
					}
					resultIStart += resultDepthSize;
					if( indexIStart != 0 ) {
						indexIStart += resultDepthSize;
					}
				}
				sourceJIndex += sourceRowSize;
			}
			resultJStart += resultRowSize;
			if( indexJStart != 0 ) {
				indexJStart += resultRowSize;
			}
		}

		sourceObject += sourceObjectSize;
	}
}

//-------------------------------------------------------------------------------------------------------
// 3d mean pooling

static void blob3dMeanPoolingProcessItem( const float* sourceObject, int sourceIndex,
	int sseChannels, int nonSseChannels, float* resultData )
{
	const float* sourceData = sourceObject + sourceIndex;

	for( int c = 0; c < sseChannels; ++c ) {
		__m128 src = _mm_loadu_ps( sourceData );
		__m128 res = _mm_loadu_ps( resultData );
		_mm_storeu_ps( resultData, _mm_add_ps( res, src ) );

		sourceData += 4;
		resultData += 4;
	}

	for( int c = 0; c < nonSseChannels; ++c ) {
		*resultData++ += *sourceData++;
	}
}

void CCpuMathEngine::Blob3dMeanPooling( const C3dMeanPoolingDesc& convDesc, const CConstFloatHandle& sourceData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommon3dMeanPoolingDesc& desc = static_cast<const CCommon3dMeanPoolingDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const float* sourceObject = GetRaw( sourceData );
	float* resultJStart = GetRaw( resultData );

	const int sourceDepthSize = source.Depth() * source.Channels();
	const int sourceRowSize = source.Width() * sourceDepthSize;
	const int sourceObjectSize = source.Height() * sourceRowSize;

	const int resultDepthSize = result.Depth() * result.Channels();
	const int resultRowSize = result.Width() * resultDepthSize;

	int sseChannels;
	int nonSseChannels;
	checkSse( result.Channels(), sseChannels, nonSseChannels );

	for( int b = 0; b < result.ObjectCount(); ++b ) {
		// Go through all "cube blocks" and then go over values in each block
		// So we get a forward pass through the source and filterHeight * filterWidth passes through the result
		for( int j = 0; j < result.Height(); ++j ) {
			int sourceJIndex = j * desc.StrideHeight * sourceRowSize;
			for( int filterJ = 0; filterJ < desc.FilterHeight; ++filterJ ) {
				float* resultIStart = resultJStart;

				for( int i = 0; i < result.Width(); ++i ) {
					int sourceIIndex = sourceJIndex + i * desc.StrideWidth * sourceDepthSize;
					for( int filterI = 0; filterI < desc.FilterWidth; ++filterI ) {
						float* resultDataPtr = resultIStart;

						for( int k = 0; k < result.Depth(); ++k ) {
							int sourceIndex = sourceIIndex + k * desc.StrideDepth * source.Channels();
							for( int filterK = 0; filterK < desc.FilterDepth; ++filterK ) {
								if( ( filterJ == 0 ) && ( filterI == 0 ) && ( filterK == 0 ) ) {
									blob3dMeanMaxPoolingProcessFirstItem( sourceObject, sourceIndex,
										sseChannels, nonSseChannels, resultDataPtr );
								} else {
									blob3dMeanPoolingProcessItem( sourceObject, sourceIndex,
										sseChannels, nonSseChannels, resultDataPtr );
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

	// Divide the result by filter volume
	CFloatHandleStackVar denom( mathEngine(), 1 );
	denom.SetValue( 1.f / desc.FilterHeight / desc.FilterWidth / desc.FilterDepth );
	VectorMultiply( resultData, resultData, result.BlobSize(), denom );
}

static void applyMeanPoolingBackwardSet( const float* resultDiff, int sseChannels, int nonSseChannels,
	int filterHeight, int filterWidth, int filterDepth, float* sourceDiff,
	int sourceRowSize, int sourceDepthSize )
{
	for( int j = 0; j < filterHeight; ++j ) {
		float* sourceDiffDepth = sourceDiff;
		for( int i = 0; i < filterWidth; ++i ) {
			float* sourceDiffPixel = sourceDiffDepth;
			for( int k = 0; k < filterDepth; ++k ) {
				const float* resultDiffPixel = resultDiff;
				for( int c = 0; c < sseChannels; ++c ) {
					_mm_storeu_ps( sourceDiffPixel, _mm_loadu_ps( resultDiffPixel ) );
					sourceDiffPixel += 4;
					resultDiffPixel += 4;
				}
				for( int c = 0; c < nonSseChannels; ++c ) {
					*sourceDiffPixel++ = *resultDiffPixel++;
				}
			}
			sourceDiffDepth += sourceDepthSize;
		}
		sourceDiff += sourceRowSize;
	}
}

static void applyMeanPoolingBackwardAdd( const float* resultDiff, int sseChannels, int nonSseChannels,
	int filterHeight, int filterWidth, int filterDepth, float* sourceDiff,
	int sourceRowSize, int sourceDepthSize )
{
	for( int j = 0; j < filterHeight; ++j ) {
		float* sourceDiffDepth = sourceDiff;
		for( int i = 0; i < filterWidth; ++i ) {
			float* sourceDiffPixel = sourceDiffDepth;
			for( int k = 0; k < filterDepth; ++k ) {
				const float* resultDiffPixel = resultDiff;
				for( int c = 0; c < sseChannels; ++c ) {
					__m128 src = _mm_loadu_ps( resultDiffPixel );
					__m128 res = _mm_loadu_ps( sourceDiffPixel );
					_mm_storeu_ps( sourceDiffPixel, _mm_add_ps( res, src ) );
					sourceDiffPixel += 4;
					resultDiffPixel += 4;
				}
				for( int c = 0; c < nonSseChannels; ++c ) {
					*sourceDiffPixel++ += *resultDiffPixel++;
				}
			}
			sourceDiffDepth += sourceDepthSize;
		}
		sourceDiff += sourceRowSize;
	}
}

void CCpuMathEngine::Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc& poolingDesc,
	const CConstFloatHandle& resultDiff, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommon3dMeanPoolingDesc& desc = static_cast<const CCommon3dMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	if( desc.FilterHeight != desc.StrideHeight || desc.FilterWidth != desc.StrideWidth || desc.FilterDepth != desc.StrideDepth ) {
		// Either the cube blocks for pooling have nonzero intersections and several diffs should be added up
		// or some of the data is skipped when pooling, and diff should be set to 0 for them
		VectorFill( sourceDiff, 0, source.BlobSize() );
	}

	// The flag that indicates that the cube blocks for pooling have nonzero intersections
	bool isIntersect = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth || desc.FilterDepth > desc.StrideDepth;

	int sseChannels;
	int nonSseChannels;
	checkSse( result.Channels(), sseChannels, nonSseChannels );

	const int sourceDepthSize = source.Depth() * source.Channels();
	const int sourceRowSize = sourceDepthSize * source.Width();
	const int sourceObjectSize = sourceRowSize * source.Height();

	const float* resultDiffPtr = GetRaw( resultDiff );
	float* sourceDiffPtr = GetRaw( sourceDiff );

	for( int b = 0; b < result.ObjectCount(); ++b ) {
		int jStart = 0;
		for( int j = 0; j < result.Height(); ++j ) {
			int iStart = jStart;
			for( int i = 0; i < result.Width(); ++i ) {
				int kStart = iStart;
				for( int k = 0; k < result.Depth(); ++k ) {
					if( isIntersect ) {
						applyMeanPoolingBackwardAdd( resultDiffPtr, sseChannels, nonSseChannels,
							desc.FilterHeight, desc.FilterWidth, desc.FilterDepth, sourceDiffPtr + kStart,
							sourceRowSize, sourceDepthSize );
					} else {
						applyMeanPoolingBackwardSet( resultDiffPtr, sseChannels, nonSseChannels,
							desc.FilterHeight, desc.FilterWidth, desc.FilterDepth, sourceDiffPtr + kStart,
							sourceRowSize, sourceDepthSize );
					}
					resultDiffPtr += result.Channels();
					kStart += source.Channels() * desc.StrideDepth;
				}
				iStart += sourceDepthSize * desc.StrideWidth;
			}
			jStart += sourceRowSize * desc.StrideHeight;
		}
		sourceDiffPtr += sourceObjectSize;
	}

	// Divide the source by the filter volume
	CFloatHandleStackVar denom( mathEngine(), 1 );
	denom.SetValue( 1.f / desc.FilterHeight / desc.FilterWidth / desc.FilterDepth );
	VectorMultiply( sourceDiff, sourceDiff, source.BlobSize(), denom );
}

//-------------------------------------------------------------------------------------------------------

void CCpuMathEngine::BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CIntHandle* maxIndices, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices == 0 || maxIndices->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonMaxOverTimePoolingDesc& desc = static_cast<const CCommonMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	if( maxIndices != 0 ) {
		int seqElemSize = source.BlobSize() / source.BatchLength();
		int seqElemSizeSse = seqElemSize / 4;
		int seqElemSizeNonSse = seqElemSize % 4;

		const float* sourceStart = GetRaw( sourceData );
		float* resultStart = GetRaw( resultData );
		int* indexStart = GetRaw( *maxIndices );

		int indexValueStart = 0;
		for( int l = 0; l < result.BatchLength(); ++l ) {
			const float* sourceDataPtr = sourceStart;

			// Set the initial values (the zero value)
			dataCopy( resultStart, sourceStart, seqElemSize );
			vectorFill( indexStart, indexValueStart, seqElemSize );

			sourceDataPtr += seqElemSize;

			for( int n = 1; n < desc.FilterLen; ++n ) {
				// Restart the result data
				float* resultDataPtr = resultStart;
				int* indexData = indexStart;
				int indexValue = indexValueStart + n;
				__m128i indexValueSse = _mm_set1_epi32( indexValue );

				for( int i = 0; i < seqElemSizeSse; ++i ) {
					__m128 src = _mm_loadu_ps( sourceDataPtr );
					__m128 res = _mm_loadu_ps( resultDataPtr );
					__m128i cmp = _mm_castps_si128( _mm_cmpgt_ps( src, res ) );

					res = _mm_max_ps( res, src );
					_mm_storeu_ps( resultDataPtr, res );

					__m128i ind = _mm_loadu_si128( ( const __m128i* )indexData );
					ind = _mm_or_si128( _mm_andnot_si128( cmp, ind ), _mm_and_si128( cmp, indexValueSse ) );
					_mm_storeu_si128( ( __m128i* )indexData, ind );

					sourceDataPtr += 4;
					resultDataPtr += 4;
					indexData += 4;
				}
				for( int i = 0; i < seqElemSizeNonSse; ++i ) {
					if( *sourceDataPtr > *resultDataPtr ) {
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
		int seqElemSize = source.BlobSize() / source.BatchLength();
		int seqElemSizeSse = seqElemSize / 4;
		int seqElemSizeNonSse = seqElemSize % 4;

		const float* sourceStart = GetRaw( sourceData );
		float* resultStart = GetRaw( resultData );

		for( int l = 0; l < result.BatchLength(); ++l ) {
			const float* sourceDataPtr = sourceStart;

			// Set the initial values (the zero value)
			dataCopy( resultStart, sourceStart, seqElemSize );

			sourceDataPtr += seqElemSize;

			for( int n = 1; n < desc.FilterLen; ++n ) {
				// Restart the result data
				float* resultDataPtr = resultStart;
				for( int i = 0; i < seqElemSizeSse; ++i ) {
					__m128 src = _mm_loadu_ps( sourceDataPtr );
					__m128 res = _mm_loadu_ps( resultDataPtr );

					res = _mm_max_ps( res, src );
					_mm_storeu_ps( resultDataPtr, res );

					sourceDataPtr += 4;
					resultDataPtr += 4;
				}

				for( int i = 0; i < seqElemSizeNonSse; ++i ) {
					if( *sourceDataPtr > *resultDataPtr ) {
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

void CCpuMathEngine::AddWidthIndex( const CBlobDesc& source, const CConstFloatHandle& sourceData, bool isForward, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* pSource = GetRaw( sourceData );
	float* pResult = GetRaw( resultData );

	for( int batch = 0; batch < source.ObjectCount(); ++batch ) {
		for( int h = 0; h < source.Height(); ++h ) {
			for( int w = 0; w < source.Width(); ++w ) {
				for( int c = 0; c < source.Channels(); ++c ) {
					*pResult = isForward ? *pSource + w : *pSource - w;
					++pSource;
					++pResult;
				}
			}
		}
	}
}

void CCpuMathEngine::AddWidthIndex( const CBlobDesc& source, const CConstIntHandle& sourceData, bool isForward, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* pSource = GetRaw( sourceData );
	int* pResult = GetRaw( resultData );

	for( int batch = 0; batch < source.ObjectCount(); ++batch ) {
		for( int h = 0; h < source.Height(); ++h ) {
			for( int w = 0; w < source.Width(); ++w ) {
				for( int c = 0; c < source.Channels(); ++c ) {
					*pResult = isForward ? *pSource + w : *pSource - w;
					++pSource;
					++pResult;
				}
			}
		}
	}
}

void CCpuMathEngine::AddHeightIndex( const CBlobDesc& source, const CConstFloatHandle& sourceData, bool isForward, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* pSource = GetRaw( sourceData );
	float* pResult = GetRaw( resultData );

	for( int batch = 0; batch < source.ObjectCount(); ++batch ) {
		for( int h = 0; h < source.Height(); ++h ) {
			for( int w = 0; w < source.Width(); ++w ) {
				for( int c = 0; c < source.Channels(); ++c ) {
					*pResult = isForward ? *pSource + h : *pSource - h;
					++pSource;
					++pResult;
				}
			}
		}
	}
}

void CCpuMathEngine::AddHeightIndex( const CBlobDesc& source, const CConstIntHandle& sourceData, bool isForward, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* pSource = GetRaw( sourceData );
	int* pResult = GetRaw( resultData );

	for( int batch = 0; batch < source.ObjectCount(); ++batch ) {
		for( int h = 0; h < source.Height(); ++h ) {
			for( int w = 0; w < source.Width(); ++w ) {
				for( int c = 0; c < source.Channels(); ++c ) {
					*pResult = isForward ? *pSource + h : *pSource - h;
					++pSource;
					++pResult;
				}
			}
		}
	}
}

} // namespace NeoML

#endif // NEOML_USE_SSE
