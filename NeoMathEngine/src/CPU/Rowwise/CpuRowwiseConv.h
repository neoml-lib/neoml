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

#pragma once

#include "CpuRowwiseInterface.h"
#include "CpuRowwiseCommon.h"
#include <CpuMathEngineDnnConv.h>
#include <CpuMathEngine.h>

namespace NeoML {

class CCpuMathEngine::CRowwiseConv : public IRowwiseCpuImpl, public CRowwiseOperationDesc {
public:
	CRowwiseConv( CCpuMathEngine& mathEngine, int inputChannels, int padH, int padW, int strideH, int strideW,
		int dilH, int dilW, int fC, int fH, int fW, const float* filter, const float* freeTerm ) :
		mathEngine( mathEngine ),
		desc( std::unique_ptr<CConvolutionDesc>(), CBlobDesc(), CBlobDesc(), CBlobDesc( { 1, fC, 1, fH, fW, 1, inputChannels } ),
			padH, padW, strideH, strideW, dilH, dilW ),
		filter( filter ),
		freeTerm( freeTerm )
	{
	}

	int MinInputRowCount() const override;

	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InOperationBufferSize() const override;
	int OutputRowCount() const override { return desc.Result.ObjectCount() * desc.Result.Height(); }
	int OutputRowSize() const override { return desc.Result.Width() * desc.Result.Channels(); }
	bool IsTrivial() const override { return false; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	CCpuMathEngine& mathEngine;
	CCpuConvolutionDesc desc;
	const float* filter;
	const float* freeTerm;

	bool is1x1Conv() const { return desc.Filter.GeometricalSize() == 1 && desc.PaddingHeight == 0
		&& desc.PaddingWidth == 0 && desc.StrideHeight == 1 && desc.StrideWidth == 1; }
	int getCacheItemCount() const;
};

inline int CCpuMathEngine::CRowwiseConv::MinInputRowCount() const 
{
	const int effectiveFilterHeight = 1 + ( desc.Filter.Height() - 1 ) * desc.DilationHeight;
	if( desc.SimdConvolutionDesc != nullptr ) {
		return effectiveFilterHeight;
	}
	// Other algorithms face significant performance reduction in matmul if first matrix is too small
	return std::max( effectiveFilterHeight, ( 64 + desc.Result.Width() - 1 ) / desc.Result.Width() );
}

inline CBlobDesc CCpuMathEngine::CRowwiseConv::Reshape( const CBlobDesc& inputSize )
{
	auto convOutputSize = [] ( int input, int filter, int padding,
		int stride, int dilation ) -> int
	{
		return 1 + ( input - ( filter - 1 ) * dilation + 2 * padding - 1 ) / stride;
	};

	CBlobDesc outputSize = inputSize;
	outputSize.SetDimSize( BD_Height, convOutputSize( inputSize.Height(), desc.Filter.Height(), desc.PaddingHeight,
		desc.StrideHeight, desc.DilationHeight ) );
	outputSize.SetDimSize( BD_Width, convOutputSize( inputSize.Width(), desc.Filter.Width(), desc.PaddingWidth,
		desc.StrideWidth, desc.DilationWidth ) );
	outputSize.SetDimSize( BD_Channels, desc.Filter.ObjectCount() );
	desc = CCpuConvolutionDesc( std::unique_ptr<CConvolutionDesc>(), inputSize, outputSize, desc.Filter,
		desc.PaddingHeight, desc.PaddingWidth, desc.StrideHeight, desc.StrideWidth,
		desc.DilationHeight, desc.DilationWidth );

	std::unique_ptr<CConvolutionDesc> simdConvolutionDesc;
	if( mathEngine.simdMathEngine != nullptr ) {
		desc.SimdConvolutionDesc = std::unique_ptr<CConvolutionDesc>( mathEngine.simdMathEngine->InitBlobConvolution(
			desc.Source, desc.PaddingHeight, desc.PaddingWidth, desc.StrideHeight, desc.StrideWidth,
			desc.DilationHeight, desc.DilationWidth, desc.Filter, desc.Result ) );
	}
	return outputSize;
}

inline int CCpuMathEngine::CRowwiseConv::InOperationBufferSize() const
{
	if( is1x1Conv() || desc.SimdConvolutionDesc != nullptr ) {
		return 0;
	}

	return getCacheItemCount() * desc.Filter.ObjectSize();
}

inline IRowwiseCpuImpl::CProcessingReport CCpuMathEngine::CRowwiseConv::Process( const float* input, int inputRowIndex,
	int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const
{
	CProcessingReport report = RowwiseConvProcessingReport( inputRowIndex, inputRowsAvailable, outputRowIndex,
		outputRowsAvailable, desc.Source.Height(), desc.Result.Height(), desc.Filter.Height(), desc.PaddingHeight,
		desc.StrideHeight, desc.DilationHeight );
	if( report.OutputRowsCalculated == 0 ) {
		PRESUME_EXPR( report.InputRowsMayBeRemoved == 0 );
		return report;
	}

	if( desc.SimdConvolutionDesc != nullptr ) {
		mathEngine.simdMathEngine->BlobConvolutionRowwise( *desc.SimdConvolutionDesc,
			input, inputRowIndex, filter, freeTerm,
			output, outputRowIndex, report.OutputRowsCalculated );
		return report;
	}

	const int inputRowSize = desc.Source.Width() * desc.Source.Channels();
	const int firstInputRowUsed = RowwiseConvFirstInputRow( outputRowIndex, desc.Source.Height(),
		desc.Result.Height(), desc.StrideHeight, desc.PaddingHeight );
	if( inputRowIndex < firstInputRowUsed ) {
		const int diff = firstInputRowUsed - inputRowIndex;
		input += diff * inputRowSize;
		inputRowIndex += diff;
		inputRowsAvailable -= diff;
	}

	if( is1x1Conv() ) {
		mathEngine.multiplyMatrixByTransposedMatrix( input, report.OutputRowsCalculated * desc.Result.Width(),
			desc.Source.Channels(), desc.Source.Channels(), filter, desc.Result.Channels(),
			desc.Source.Channels(), output, desc.Result.Channels() );
		if( freeTerm != nullptr ) {
			mathEngine.addVectorToMatrixRows( output, output, report.OutputRowsCalculated * desc.Result.Width(),
				desc.Result.Channels(), desc.Result.Channels(), desc.Result.Channels(),
				freeTerm );
		}
		return report;
	}

	const int filterObjectCount = desc.Filter.ObjectCount();
	const int filterObjectSize = desc.Filter.ObjectSize();
	const int cacheItemCount = getCacheItemCount();

	const int imageStartOffset = inputRowIndex * desc.Source.Width() * desc.Source.Channels();

	int index = 0;
	const int count = report.OutputRowsCalculated * desc.Result.Width();
	const int start = outputRowIndex * desc.Result.Width();
	while( index < count ) {
		const int size = std::min( count - index, cacheItemCount );

		mathEngine.fillTempData( input - imageStartOffset, buffer, desc, start + index, size );

		float* resultDataPtr = output + index * filterObjectCount;

		mathEngine.multiplyMatrixByTransposedMatrix( buffer, size, filterObjectSize,
			filterObjectSize, filter, filterObjectCount, filterObjectSize, resultDataPtr,
			filterObjectCount );

		if( freeTerm!= nullptr ) {
			mathEngine.addVectorToMatrixRows( resultDataPtr, resultDataPtr, size, filterObjectCount, filterObjectCount,
				filterObjectCount, freeTerm );
		}

		index += size;
	}

	return report;
}

inline int CCpuMathEngine::CRowwiseConv::getCacheItemCount() const
{
	const int resultItemCount = desc.Result.Width() * desc.Result.Height();
	return std::max( 1, std::min( ceilTo( BlobConvolutionCacheSize / desc.Filter.ObjectSize(), 16 ),
		resultItemCount ) );
}

} // namespace NeoML
