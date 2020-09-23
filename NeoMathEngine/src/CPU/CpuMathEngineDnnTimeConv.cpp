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

#include <CpuMathEngine.h>
#include <CpuMathEngineOmp.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnConv.h>
#include <CpuMathEnginePrivate.h>

namespace NeoML {

CTimeConvolutionDesc* CCpuMathEngine::InitTimeConvolution( const CBlobDesc& source,
	int stride, int padding, int dilation, const CBlobDesc& filter, const CBlobDesc& result )
{
	ASSERT_EXPR( stride > 0 );
	ASSERT_EXPR( padding >= 0 );
	ASSERT_EXPR( dilation > 0 );
	ASSERT_EXPR( filter.BatchLength() == 1 );
	ASSERT_EXPR( filter.Width() == 1 );
	ASSERT_EXPR( filter.Depth() == 1 );
	ASSERT_EXPR( filter.Channels() == source.ObjectSize() );
	ASSERT_EXPR( source.BatchLength() + 2 * padding >= ( filter.Height() - 1 ) * dilation + 1 );
	ASSERT_EXPR( result.BatchLength() == ( source.BatchLength() - ( filter.Height() - 1 ) * dilation - 1 + 2 * padding ) / stride + 1 );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.ListSize() == 1 && source.ListSize() == 1 );
	ASSERT_EXPR( result.Width() == 1 );
	ASSERT_EXPR( result.Height() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( result.Channels() == filter.BatchWidth() );
	ASSERT_EXPR( padding < ( filter.Height() - 1 ) * dilation + 1 );

	CCommonTimeConvolutionDesc* desc = new CCommonTimeConvolutionDesc( source, filter, result, stride, padding, dilation );
	return desc;
}

void CCpuMathEngine::BlobTimeConvolution( const CTimeConvolutionDesc& convDesc, const CFloatHandle& sourceData,
	const CFloatHandle& filterData, const CFloatHandle& freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const float* sourceDataRaw = GetRaw( sourceData );
	const float* filterDataRaw = GetRaw( filterData );
	float* resultDataRaw = GetRaw( resultData );

	const CCommonTimeConvolutionDesc& desc = static_cast<const CCommonTimeConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;
	const CBlobDesc& filter = desc.Filter;

	int filterDataSize = filter.Height() * filter.Channels();
	int outputObjectSize = result.ObjectSize();
	int inputObjectSize = source.ObjectSize();
	int inputRowSize = source.BatchWidth() * inputObjectSize;

	const int curThreadCount = IsOmpRelevant( result.BatchLength() ) ? threadCount : 1;

	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int outSeqNum = 0; outSeqNum < result.BatchLength(); ++outSeqNum ) {
		int filterRowStart = 0;
		int inputRowStart = outSeqNum * desc.Stride - desc.Padding;
		if( inputRowStart < 0 ) {
			filterRowStart = ( -inputRowStart - 1 ) / desc.Dilation + 1;
			inputRowStart = inputRowStart + filterRowStart * desc.Dilation;
		}
		int filterRowCount = filter.Height() - filterRowStart;

		if( inputRowStart + ( filterRowCount - 1 ) * desc.Dilation >= source.BatchLength() ) {
			filterRowCount = ( source.BatchLength() - inputRowStart + desc.Dilation - 1 ) / desc.Dilation;
		}

		float* outputPtr = resultDataRaw + outSeqNum * result.BatchWidth() * outputObjectSize;
		const float* inputPtr = sourceDataRaw + inputRowStart * inputRowSize;
		const float* filterPtr = filterDataRaw + filterRowStart * filter.Channels();

		multiplyMatrixByTransposedMatrix( inputPtr,
			source.BatchWidth(), inputObjectSize, inputObjectSize,
			filterPtr, filter.BatchWidth(), filterDataSize,
			outputPtr, outputObjectSize, outputObjectSize * source.BatchWidth() );

		for( int i = 1; i < filterRowCount; ++i ) {
			inputPtr += inputRowSize * desc.Dilation;
			filterPtr += filter.Channels();

			multiplyMatrixByTransposedMatrixAndAdd( inputPtr, source.BatchWidth(), inputObjectSize, inputObjectSize,
				filterPtr, filter.BatchWidth(), filterDataSize, outputPtr, outputObjectSize );
		}
	}

	AddVectorToMatrixRows( 1, resultData, resultData, result.ObjectCount(), result.ObjectSize(), freeTermData );
}

void CCpuMathEngine::BlobTimeConvolutionBackward( const CTimeConvolutionDesc& convDesc, const CFloatHandle& outputDiffData,
	const CFloatHandle& filterData, const CFloatHandle& freeTermData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData.GetMathEngine() == this );
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );

	const float* outputDiffDataRaw = GetRaw( outputDiffData );
	const float* filterDataRaw = GetRaw( filterData );
	float* inputDiffDataRaw = GetRaw( inputDiffData );

	const CCommonTimeConvolutionDesc& desc = static_cast<const CCommonTimeConvolutionDesc&>( convDesc );
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& filter = desc.Filter;

	int filterDataSize = filter.Height() * filter.Channels();
	int inputObjectSize = inputDiff.ObjectSize();
	int inputRowSize = inputDiff.BatchWidth() * inputObjectSize;
	int outputObjectSize = outputDiff.ObjectSize();
	int outputRowSize = outputDiff.BatchWidth() * outputObjectSize;

	const int curThreadCount = IsOmpRelevant( inputDiff.BatchLength() ) ? threadCount : 1;

	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int inSeqNum = 0; inSeqNum < inputDiff.BatchLength(); ++inSeqNum ) {
		float* inputDiffDataPtr = inputDiffDataRaw + inSeqNum * inputRowSize;
		vectorFill0( inputDiffDataPtr, inputObjectSize * inputDiff.BatchWidth() );

		for( int filterRow = 0; filterRow < filter.Height(); filterRow++ ) {
			int inSeqNumFirst = inSeqNum - filterRow * desc.Dilation;
			if( inSeqNumFirst < -desc.Padding ) {
				break; // the next values can only be smaller
			}
			if( ( inSeqNumFirst + desc.Padding ) % desc.Stride != 0 ) {
				continue; // this filter row not applicable to the current row
			}
			int outSeqNum = ( inSeqNumFirst + desc.Padding ) / desc.Stride;
			if( outSeqNum >= outputDiff.BatchLength() ) {
				continue;
			}

			const float* outputDiffPtr = outputDiffDataRaw + outSeqNum * outputRowSize;
			const float* filterPtr = filterDataRaw + filterRow * filter.Channels();

			multiplyMatrixByMatrixAndAdd( outputDiffPtr,
				outputDiff.BatchWidth(), outputObjectSize, outputObjectSize,
				filterPtr, filter.Channels(), filterDataSize,
				inputDiffDataPtr, inputObjectSize, inputObjectSize * inputDiff.BatchWidth() );
		}
	}
}

void CCpuMathEngine::BlobTimeConvolutionLearnAdd( const CTimeConvolutionDesc& convDesc, const CFloatHandle& inputData,
	const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle& freeTermDiffData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( filterDiffData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermDiffData.GetMathEngine() == this );
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );

	const float* outputDiffDataRaw = GetRaw( outputDiffData );
	const float* inputDataRaw = GetRaw( inputData );

	const CCommonTimeConvolutionDesc& desc = static_cast<const CCommonTimeConvolutionDesc&>( convDesc );
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& filterDiff = desc.Filter;

	// Train the filter
	int filterDataSize = filterDiff.Height() * filterDiff.Channels();

	COmpReduction1DData filterDiffItem( mathEngine(), filterDiffData, filterDiff.BlobSize() );
	COmpReduction<COmpReduction1DData> ompReduction( threadCount, filterDiffItem );

	const int curThreadCount = IsOmpRelevant( outputDiff.BatchLength() ) ? threadCount : 1;

	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int outSeqNum = 0; outSeqNum < outputDiff.BatchLength(); ++outSeqNum ) {
		const float* outputDiffPtr = outputDiffDataRaw +
			outSeqNum * outputDiff.BatchWidth() * outputDiff.ObjectSize();
		float* ompReductionPrivatePtr = GetRaw( ompReduction.GetPrivate().Data );

		for( int filterRow = 0; filterRow < filterDiff.Height(); ++filterRow ) {
			int inSeqNum = outSeqNum * desc.Stride - desc.Padding + filterRow * desc.Dilation;
			if( inSeqNum < 0 || inSeqNum >= input.BatchLength() ) {
				continue; // padding or went out of the input bounds
			}

			const float* inputPtr = inputDataRaw + inSeqNum * input.BatchWidth() * filterDiff.Channels();
			float* filterDiffPtr = ompReductionPrivatePtr + filterRow * filterDiff.Channels();

			multiplyTransposedMatrixByMatrixAndAdd( outputDiffPtr,
				outputDiff.BatchWidth(), filterDiff.BatchWidth(), filterDiff.BatchWidth(),
				inputPtr, filterDiff.Channels(), filterDiff.Channels(),
				filterDiffPtr, filterDataSize, filterDiff.BlobSize() - filterRow * filterDiff.Channels() );
		}
	}

	ompReduction.Reduce();

	// Train the free term
	SumMatrixRowsAdd( 1, freeTermDiffData, outputDiffData, outputDiff.ObjectCount(), filterDiff.ObjectCount() );
}

} // namespace NeoML
