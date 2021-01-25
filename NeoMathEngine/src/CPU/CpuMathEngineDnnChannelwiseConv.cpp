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
#include <CpuMathEngine.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnConv.h>
#include <CpuMathEnginePrivate.h>

namespace NeoML {

static inline void fillResultRow( const CCommonChannelwiseConvolutionDesc& desc, const float* freeTerm, float* result )
{
	const int channels = desc.Result.Channels();
	int count = desc.Result.Width();
	while( count > 0 ) {
		NeoML::dataCopy( result, freeTerm, channels );
		result += channels;
		count--;
	}
}

static inline void processFilterRowStride1( const CCommonChannelwiseConvolutionDesc& desc, const float* filter, const float* source, float* result )
{
	PRESUME_EXPR( desc.PaddingWidth == 1 );
	PRESUME_EXPR( desc.Filter.Width() == 3 );

	const int resultWidth = desc.Result.Width();
	int width = resultWidth - 2;
	int channels = desc.Result.Channels();
	const float* filter1 = filter + channels;
	const float* filter2 = filter1 + channels;

	NeoML::vectorEltwiseMultiplyAdd( filter1, source, result, channels );
	if( resultWidth > 1 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter2, source + channels, result, channels );
	}

	const float* sourcePos = source;
	float* resultPos = result + channels;
	while( width >= 2 ) {
		channelwise1x3( sourcePos, filter, filter1, filter2, resultPos, channels );

		resultPos += 2 * channels;
		sourcePos += 2 * channels;
		width -= 2;
	}
	
	while( width > 0 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + channels, sourcePos + channels, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + 2 * channels, sourcePos + 2 * channels, resultPos, channels );
		resultPos += channels;
		sourcePos += channels;
		width--;
	}

	if( resultWidth > 1 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + channels, sourcePos + channels, resultPos, channels );
	}
}

static inline void processFilterRowStride2( const CCommonChannelwiseConvolutionDesc& desc, const float* filter, const float* source, float* result )
{
	PRESUME_EXPR( desc.PaddingWidth == 1 );
	PRESUME_EXPR( desc.Filter.Width() == 3 );

	const int resultWidth = desc.Result.Width();
	int width = resultWidth - 1 - ( desc.Source.Width() % 2 );
	int channels = desc.Result.Channels();
	const float* filter1 = filter + channels;
	const float* filter2 = filter1 + channels;

	NeoML::vectorEltwiseMultiplyAdd( filter1, source, result, channels );
	if( resultWidth > 1 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter2, source + channels, result, channels );
	}

	const float* sourcePos = source + channels;
	float* resultPos = result + channels;

	while( width > 0 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + channels, sourcePos + channels, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + 2 * channels, sourcePos + 2 * channels, resultPos, channels );
		resultPos += channels;
		sourcePos += 2 * channels;
		width--;
	}

	if( desc.Source.Width() % 2 == 1 && resultWidth > 1 ) {
		NeoML::vectorEltwiseMultiplyAdd( filter, sourcePos, resultPos, channels );
		NeoML::vectorEltwiseMultiplyAdd( filter + channels, sourcePos + channels, resultPos, channels );
	}
}

void CCpuMathEngine::blobChannelwiseConvolutionFilter3x3Padding1Stride2( const CCommonChannelwiseConvolutionDesc& desc,
	const float* source, const float* filter, const float* freeTerm, float* result )
{
	const CBlobDesc& sourceDesc = desc.Source;
	const CBlobDesc& filterDesc = desc.Filter;
	const CBlobDesc& resultDesc = desc.Result;

	const int curThreadCount = IsOmpRelevant( sourceDesc.ObjectCount() * resultDesc.Height(),
		static_cast<int64_t>( sourceDesc.BlobSize() ) * filterDesc.BlobSize() ) ? threadCount : 1;

	const int channels = sourceDesc.Channels() * sourceDesc.Depth();

	const int inputRowSize = sourceDesc.Width() * channels;
	const int outputRowSize = resultDesc.Width() * channels;
	const int filterRowSize = filterDesc.Width() * channels;

	const int inputObjectSize = inputRowSize * sourceDesc.Height();
	const int outputObjectSize = outputRowSize * resultDesc.Height();

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int batchStart;
		int batchCount;
		int resultStart;
		int resultCount;
		if( OmpGetTaskIndexAndCount2D( sourceDesc.ObjectCount(), resultDesc.Height(), batchStart, batchCount, resultStart, resultCount ) ) {
			const float* src = source + batchStart * inputObjectSize;
			const float* srcEnd = src + batchCount * inputObjectSize;
			float* resultFirstRow = 0;
			const float* sourceFirstRow = 0;
			float* resultLastRow = 0;
			const float* sourceLastRow = 0;
			if( resultStart == 0 ) {
				resultFirstRow = result + batchStart * outputObjectSize;
				sourceFirstRow = src;
				resultStart++;
				resultCount--;
			}
			if( resultStart + resultCount == resultDesc.Height() && sourceDesc.Height() % 2 == 1 ) {
				resultLastRow = result + batchStart * outputObjectSize + ( resultDesc.Height() - 1 ) * outputRowSize;
				sourceLastRow = src + ( sourceDesc.Height() - 2 ) * inputRowSize;
				resultCount--;
			}

			float* resultRow = result + batchStart * outputObjectSize + resultStart * outputRowSize;
			float* resultRowEnd = resultRow + resultCount * outputRowSize;
			for( ; src < srcEnd; src += inputObjectSize ) {
				const float* srcRow = src + ( resultStart * 2 - 1 ) * inputRowSize;
				float* resRow = resultRow;

				if( resultFirstRow != 0 ) {
					if( freeTerm != 0 ) {
						fillResultRow( desc, freeTerm, resultFirstRow );
					} else {
						NeoML::vectorFill( resultFirstRow, 0, resultDesc.Width() * channels );
					}

					processFilterRowStride2( desc, filter + filterRowSize, sourceFirstRow, resultFirstRow );
					if( resultCount >= 0 ) {
						processFilterRowStride2( desc, filter + 2 * filterRowSize, sourceFirstRow + inputRowSize, resultFirstRow );
					}
					resultFirstRow += outputObjectSize;
					sourceFirstRow += inputObjectSize;
				}

				for( ; resRow < resultRowEnd; resRow += outputRowSize, srcRow += 2 * inputRowSize ) {
					if( freeTerm != 0 ) {
						fillResultRow( desc, freeTerm, resRow );
					} else {
						NeoML::vectorFill( resRow, 0, resultDesc.Width() * channels );
					}

					processFilterRowStride2( desc, filter, srcRow, resRow );
					processFilterRowStride2( desc, filter + filterRowSize, srcRow + inputRowSize, resRow );
					processFilterRowStride2( desc, filter + 2 * filterRowSize, srcRow + 2 * inputRowSize, resRow );
				}

				if( resultLastRow != 0 && resultCount >= 0 ) {
					if( freeTerm != 0 ) {
						fillResultRow( desc, freeTerm, resultLastRow );
					} else {
						NeoML::vectorFill( resultLastRow, 0, resultDesc.Width() * channels );
					}

					processFilterRowStride2( desc, filter, sourceLastRow, resultLastRow );
					processFilterRowStride2( desc, filter + filterRowSize, sourceLastRow + inputRowSize, resultLastRow );
					resultLastRow += outputObjectSize;
					sourceLastRow += inputObjectSize;
				}

				resultRow += outputObjectSize;
				resultRowEnd += outputObjectSize;
			}
		}
	}
}

void CCpuMathEngine::blobChannelwiseConvolutionFilter3x3Padding1Stride1( const CCommonChannelwiseConvolutionDesc& desc,
	const float* source, const float* filter, const float* freeTerm, float* result )
{
	const CBlobDesc& sourceDesc = desc.Source;
	const CBlobDesc& filterDesc = desc.Filter;
	const CBlobDesc& resultDesc = desc.Result;

	const int curThreadCount = IsOmpRelevant( sourceDesc.ObjectCount() * resultDesc.Height(),
		static_cast<int64_t>( sourceDesc.BlobSize() ) * filterDesc.BlobSize() ) ? threadCount : 1;

	const int channels = sourceDesc.Channels() * sourceDesc.Depth();

	const int inputRowSize = sourceDesc.Width() * channels;
	const int outputRowSize = resultDesc.Width() * channels;
	const int filterRowSize = filterDesc.Width() * channels;

	const int inputObjectSize = inputRowSize * sourceDesc.Height();
	const int outputObjectSize = outputRowSize * resultDesc.Height();

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int batchStart;
		int batchCount;
		int resultStart;
		int resultCount;
		if( OmpGetTaskIndexAndCount2D( sourceDesc.ObjectCount(), resultDesc.Height(), batchStart, batchCount, resultStart, resultCount ) ) {
			const float* src = source + batchStart * inputObjectSize;
			const float* srcEnd = src + batchCount * inputObjectSize;
			float* resultFirstRow = 0;
			const float* sourceFirstRow = 0;
			float* resultLastRow = 0;
			const float* sourceLastRow = 0;
			if( resultStart == 0 ) {
				resultFirstRow = result + batchStart * outputObjectSize;
				sourceFirstRow = src;
				resultStart++;
				resultCount--;
			}
			if( resultStart + resultCount == resultDesc.Height() ) {
				resultLastRow = result + batchStart * outputObjectSize + ( resultDesc.Height() - 1 ) * outputRowSize;
				sourceLastRow = src + ( resultDesc.Height() - 2 ) * inputRowSize;
				resultCount--;
			}

			float* resultRow = result + batchStart * outputObjectSize + resultStart * outputRowSize;
			float* resultRowEnd = resultRow + resultCount * outputRowSize;
			for( ; src < srcEnd; src += inputObjectSize ) {
				const float* srcRow = src + ( resultStart - 1 ) * inputRowSize;
				float* resRow = resultRow;

				if( resultFirstRow != 0 ) {
					if( freeTerm != 0 ) {
						fillResultRow( desc, freeTerm, resultFirstRow );
					} else {
						NeoML::vectorFill( resultFirstRow, 0, resultDesc.Width() * channels );
					}
					processFilterRowStride1( desc, filter + filterRowSize, sourceFirstRow, resultFirstRow );
					if( resultCount >= 0 ) {
						processFilterRowStride1( desc, filter + 2 * filterRowSize, sourceFirstRow + inputRowSize, resultFirstRow );
					}
					resultFirstRow += outputObjectSize;
					sourceFirstRow += inputObjectSize;
				}

				for( ; resRow < resultRowEnd; resRow += outputRowSize, srcRow += inputRowSize ) {
					if( freeTerm != 0 ) {
						fillResultRow( desc, freeTerm, resRow );
					} else {
						NeoML::vectorFill( resRow, 0, resultDesc.Width() * channels );
					}

					processFilterRowStride1( desc, filter, srcRow, resRow );
					processFilterRowStride1( desc, filter + filterRowSize, srcRow + inputRowSize, resRow );
					processFilterRowStride1( desc, filter + 2 * filterRowSize, srcRow + 2 * inputRowSize, resRow );
				}

				if( resultLastRow != 0 && resultCount >= 0 ) {
					if( freeTerm != 0 ) {
						fillResultRow( desc, freeTerm, resultLastRow );
					} else {
						NeoML::vectorFill( resultLastRow, 0, resultDesc.Width() * channels );
					}

					processFilterRowStride1( desc, filter, sourceLastRow, resultLastRow );
					processFilterRowStride1( desc, filter + filterRowSize, sourceLastRow + inputRowSize, resultLastRow );
					resultLastRow += outputObjectSize;
					sourceLastRow += inputObjectSize;
				}

				resultRow += outputObjectSize;
				resultRowEnd += outputObjectSize;
			}
		}
	}
}

void CCpuMathEngine::BlobChannelwiseConvolution( const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );

	const float* source = GetRaw( sourceData );
	const float* filter = GetRaw( filterData );
	const float* freeTerm = freeTermData != 0 ? GetRaw( *freeTermData ) : 0;
	float* result = GetRaw( resultData );

	if( desc.Filter.Height() == 3 && desc.Filter.Width() == 3 && desc.PaddingHeight == 1 && desc.PaddingWidth == 1 && desc.Filter.Channels() % 4 == 0 ) {
		if( desc.StrideHeight == 1 && desc.StrideWidth == 1 ) {
			blobChannelwiseConvolutionFilter3x3Padding1Stride1( desc, source, filter, freeTerm, result );
			Activation( *desc.Activation, resultData, resultData, desc.Result.BlobSize() );
			return;
		}
		if( desc.StrideHeight == 2 && desc.StrideWidth == 2 ) {
			blobChannelwiseConvolutionFilter3x3Padding1Stride2( desc, source, filter, freeTerm, result );
			Activation( *desc.Activation, resultData, resultData, desc.Result.BlobSize() );
			return;
		}
	}

	const CBlobDesc& sourceDesc = desc.Source;
	const CBlobDesc& filterDesc = desc.Filter;
	const CBlobDesc& resultDesc = desc.Result;

	const int curThreadCount = IsOmpRelevant( sourceDesc.ObjectCount() * resultDesc.Height(),
		static_cast<int64_t>( sourceDesc.BlobSize() ) * filterDesc.BlobSize() ) ? threadCount : 1;

	const int channels = sourceDesc.Channels() * sourceDesc.Depth();

	const int inputRowSize = sourceDesc.Width() * channels;
	const int outputRowSize = resultDesc.Width() * channels;
	const int filterRowSize = filterDesc.Width() * channels;

	const int inputObjectSize = inputRowSize * sourceDesc.Height();
	const int outputObjectSize = outputRowSize * resultDesc.Height();

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int batchStart;
		int batchCount;
		int resultStart;
		int resultCount;
		if( OmpGetTaskIndexAndCount2D( sourceDesc.ObjectCount(), resultDesc.Height(), batchStart, batchCount, resultStart, resultCount ) ) {
			const float* src = source + batchStart * inputObjectSize;
			const float* srcEnd = src + batchCount * inputObjectSize;
			float* res = result + batchStart * outputObjectSize + resultStart * resultDesc.Width() * channels;

			for( ; src < srcEnd; src += inputObjectSize, res += outputObjectSize ) {
				float* resultRow = res;
				float* resultRowEnd = resultRow + resultCount * outputRowSize;
				int firstFilteredRow = resultStart * desc.StrideHeight - desc.PaddingHeight;
				for( ; resultRow < resultRowEnd; resultRow += outputRowSize, firstFilteredRow += desc.StrideHeight ) {
					if( freeTerm != 0 ) {
						float* rowStart = resultRow;
						float* rowEnd = rowStart + channels * resultDesc.Width();
						for( ; rowStart < rowEnd; rowStart += channels ) {
							NeoML::dataCopy(rowStart, freeTerm, channels);
						}
					} else {
						NeoML::vectorFill(resultRow, 0, resultDesc.Width() * channels);
					}

					const int filterFirstRow = max( 0, -firstFilteredRow );
					const int filterLastRow = min( filterDesc.Height(), sourceDesc.Height() - firstFilteredRow );
					const float* filterRow = filter + filterFirstRow * filterRowSize;
					const float* filterRowEnd = filter + filterLastRow * filterRowSize;
					const float* srcRow = src + (firstFilteredRow + filterFirstRow) * inputRowSize;

					for( ; filterRow < filterRowEnd; filterRow += filterRowSize, srcRow += inputRowSize ) {
						int firstFilteredCol = -desc.PaddingWidth;
						float* resultPos = resultRow;
						float* resultPosEnd = resultPos + channels * resultDesc.Width();
						for( ; resultPos < resultPosEnd; resultPos += channels, firstFilteredCol += desc.StrideWidth ) {
							const int filterFirstCol = max( 0, -firstFilteredCol );
							const int filterLastCol = min( filterDesc.Width(), sourceDesc.Width() - firstFilteredCol );
							const float* filterPos = filterRow + filterFirstCol * channels;
							const float* filterPosEnd = filterRow + filterLastCol * channels;
							const float* srcPos = srcRow + (firstFilteredCol + filterFirstCol) * channels;
							for( ; filterPos < filterPosEnd; filterPos += channels, srcPos += channels ) {
								NeoML::vectorEltwiseMultiplyAdd(filterPos, srcPos, resultPos, channels);
							}
						}
					}
				}
			}
		}
	}

	Activation( *desc.Activation, resultData, resultData, desc.Result.BlobSize() );
}

} // namespace NeoML
