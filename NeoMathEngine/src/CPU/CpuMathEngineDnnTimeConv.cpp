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

#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnConv.h>
#include <CpuMathEnginePrivate.h>

namespace NeoML {

CTimeConvolutionDesc* CCpuMathEngine::InitTimeConvolution( const CBlobDesc& source,
	int stride, int paddingFront, int paddingBack, int dilation, const CBlobDesc& filter, const CBlobDesc& result )
{
	ASSERT_EXPR( stride > 0 );
	ASSERT_EXPR( paddingFront >= 0 );
	ASSERT_EXPR( paddingBack >= 0 );
	ASSERT_EXPR( dilation > 0 );
	ASSERT_EXPR( filter.BatchLength() == 1 );
	ASSERT_EXPR( filter.Width() == 1 );
	ASSERT_EXPR( filter.Depth() == 1 );
	ASSERT_EXPR( filter.Channels() == source.ObjectSize() );
	ASSERT_EXPR( source.BatchLength() + paddingFront + paddingBack >= ( filter.Height() - 1 ) * dilation + 1 );
	ASSERT_EXPR( result.BatchLength() == ( source.BatchLength() - ( filter.Height() - 1 ) * dilation - 1 + paddingFront + paddingBack ) / stride + 1 );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.ListSize() == 1 && source.ListSize() == 1 );
	ASSERT_EXPR( result.Width() == 1 );
	ASSERT_EXPR( result.Height() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( result.Channels() == filter.BatchWidth() );
	ASSERT_EXPR( paddingFront < ( filter.Height() - 1 ) * dilation + 1 );
	ASSERT_EXPR( paddingBack < ( filter.Height() - 1 ) * dilation + 1 );

	CCommonTimeConvolutionDesc* desc = new CCommonTimeConvolutionDesc( *this, source, result, filter, stride, paddingFront, paddingBack, dilation );
	return desc;
}

void CCpuMathEngine::BlobTimeConvolution( const CTimeConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle& freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* const sourceDataRaw = GetRaw( sourceData );
	const float* const filterDataRaw = GetRaw( filterData );
	float* const resultDataRaw = GetRaw( resultData );

	const CCommonTimeConvolutionDesc& desc = static_cast<const CCommonTimeConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;
	const CBlobDesc& filter = desc.Filter;

	const int outputObjectSize = result.ObjectSize();
	const int firstHeight = source.BatchWidth();
	const int firstWidth = source.ObjectSize();
	const int secondHeight = filter.BatchWidth();
	const int secondWidth = filter.Height() * filter.Channels();
	const int resultWidth = outputObjectSize;
	const int inputRowSize = source.BatchWidth() * firstWidth;

	for( int outSeqNum = 0; outSeqNum < result.BatchLength(); ++outSeqNum ) {
		int filterRowStart = 0;
		int inputRowStart = outSeqNum * desc.Stride - desc.PaddingFront;
		if( inputRowStart < 0 ) {
			filterRowStart = ( -inputRowStart - 1 ) / desc.Dilation + 1;
			inputRowStart = inputRowStart + filterRowStart * desc.Dilation;
		}
		int filterRowCount = filter.Height() - filterRowStart;

		if( inputRowStart + ( filterRowCount - 1 ) * desc.Dilation >= source.BatchLength() ) {
			filterRowCount = ( source.BatchLength() - inputRowStart + desc.Dilation - 1 ) / desc.Dilation;
		}

		float* const outputPtr = resultDataRaw + outSeqNum * result.BatchWidth() * outputObjectSize;
		const float* inputPtr = sourceDataRaw + inputRowStart * inputRowSize;
		const float* filterPtr = filterDataRaw + filterRowStart * filter.Channels();

		auto mulDesc = desc.smallMatricesMultiplyDescs.Get( CCommonTimeConvolutionDesc::TSMMD_Forward_MxMT,
			firstHeight, firstWidth, secondWidth, resultWidth );
		auto mulDescAdd = desc.smallMatricesMultiplyDescs.Get( CCommonTimeConvolutionDesc::TSMMD_Forward_MxMT_Add,
			firstHeight, firstWidth, secondWidth, resultWidth, /*resultAdd*/true );

		multiplyMatrixByTransposedMatrix( /*first*/inputPtr, firstHeight, firstWidth, firstWidth,
			/*second*/filterPtr, secondHeight, secondWidth, /*result*/outputPtr, resultWidth, mulDesc );

		for( int i = 1; i < filterRowCount; ++i ) {
			inputPtr += inputRowSize * desc.Dilation;
			filterPtr += filter.Channels();

			multiplyMatrixByTransposedMatrixAndAdd( /*first*/inputPtr, firstHeight, firstWidth, firstWidth,
				/*second*/filterPtr, secondHeight, secondWidth, /*result*/outputPtr, resultWidth, mulDescAdd );
		}
	}

	AddVectorToMatrixRows( /*batchSize*/1, resultData, resultData, result.ObjectCount(), result.ObjectSize(), freeTermData );
}

void CCpuMathEngine::BlobTimeConvolutionBackward( const CTimeConvolutionDesc& convDesc, const CConstFloatHandle& outputDiffData,
	const CConstFloatHandle& filterData, const CConstFloatHandle& freeTermData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData.GetMathEngine() == this );
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* const outputDiffDataRaw = GetRaw( outputDiffData );
	const float* const filterDataRaw = GetRaw( filterData );
	float* const inputDiffDataRaw = GetRaw( inputDiffData );

	const CCommonTimeConvolutionDesc& desc = static_cast<const CCommonTimeConvolutionDesc&>( convDesc );
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& filter = desc.Filter;

	const int firstHeight = outputDiff.BatchWidth();
	const int firstWidth = outputDiff.ObjectSize();
	const int secondWidth = filter.Channels();
	const int secondRowSize = filter.Height() * filter.Channels();
	const int resultWidth = inputDiff.ObjectSize();
	auto mulDesc = desc.smallMatricesMultiplyDescs.Get( CCommonTimeConvolutionDesc::TSMMD_Backward_MxM_Add,
		firstHeight, firstWidth, secondWidth, secondRowSize, resultWidth,
		/*resultAdd*/true, /*trans1*/false, /*trans2*/false  );

	const int inputRowSize = inputDiff.BatchWidth() * resultWidth;
	const int outputRowSize = firstHeight * firstWidth;

	for( int inSeqNum = 0; inSeqNum < inputDiff.BatchLength(); ++inSeqNum ) {
		float* const inputDiffDataPtr = inputDiffDataRaw + inSeqNum * inputRowSize;
		vectorFill0( inputDiffDataPtr, inputRowSize );

		for( int filterRow = 0; filterRow < filter.Height(); ++filterRow ) {
			const int inSeqNumFirst = inSeqNum - filterRow * desc.Dilation;
			if( inSeqNumFirst < -desc.PaddingFront ) {
				break; // the next values can only be smaller
			}
			if( ( inSeqNumFirst + desc.PaddingFront ) % desc.Stride != 0 ) {
				continue; // this filter row not applicable to the current row
			}
			const int outSeqNum = ( inSeqNumFirst + desc.PaddingFront ) / desc.Stride;
			if( outSeqNum >= outputDiff.BatchLength() ) {
				continue;
			}

			const float* const outputDiffPtr = outputDiffDataRaw + outSeqNum * outputRowSize;
			const float* const filterPtr = filterDataRaw + filterRow * filter.Channels();

			multiplyMatrixByMatrixAndAdd( /*first*/outputDiffPtr, firstHeight, firstWidth, firstWidth,
				/*second*/filterPtr, secondWidth, secondRowSize, /*result*/inputDiffDataPtr, resultWidth, mulDesc );
		}
	}
}

void CCpuMathEngine::BlobTimeConvolutionLearnAdd( const CTimeConvolutionDesc& convDesc, const CConstFloatHandle& inputData,
	const CConstFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle& freeTermDiffData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( filterDiffData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermDiffData.GetMathEngine() == this );
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* const outputDiffDataRaw = GetRaw( outputDiffData );
	const float* const inputDataRaw = GetRaw( inputData );
	float* const filterDiffDataRaw = GetRaw( filterDiffData );

	const CCommonTimeConvolutionDesc& desc = static_cast<const CCommonTimeConvolutionDesc&>( convDesc );
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& filterDiff = desc.Filter;

	// Train the filter
	const int firstHeight = outputDiff.BatchWidth();
	const int firstWidth = filterDiff.BatchWidth();
	const int secondWidth = filterDiff.Channels();
	const int resultWidth = filterDiff.Height() * filterDiff.Channels();
	auto mulDesc = desc.smallMatricesMultiplyDescs.Get( CCommonTimeConvolutionDesc::TSMMD_Learn_MTxM_Add,
		firstHeight, firstWidth, secondWidth, resultWidth, /*resultAdd*/true, /*trans1*/true, /*trans2*/false );

	for( int outSeqNum = 0; outSeqNum < outputDiff.BatchLength(); ++outSeqNum ) {
		const float* const outputDiffPtr = outputDiffDataRaw + outSeqNum * firstHeight * outputDiff.ObjectSize();

		for( int filterRow = 0; filterRow < filterDiff.Height(); ++filterRow ) {
			const int inSeqNum = outSeqNum * desc.Stride - desc.PaddingFront + filterRow * desc.Dilation;
			if( inSeqNum < 0 || inSeqNum >= input.BatchLength() ) {
				continue; // padding or went out of the input bounds
			}

			const float* const inputPtr = inputDataRaw + inSeqNum * input.BatchWidth() * filterDiff.Channels();
			float* const filterDiffPtr = filterDiffDataRaw + filterRow * filterDiff.Channels();

			multiplyTransposedMatrixByMatrixAndAdd( /*first*/outputDiffPtr, firstHeight, firstWidth, firstWidth,
				/*second*/inputPtr, secondWidth, secondWidth, /*result*/filterDiffPtr, resultWidth, mulDesc );
		}
	}
	// Train the free term
	SumMatrixRowsAdd( /*batchSize*/1, freeTermDiffData, outputDiffData, outputDiff.ObjectCount(), filterDiff.ObjectCount() );
}

} // namespace NeoML
