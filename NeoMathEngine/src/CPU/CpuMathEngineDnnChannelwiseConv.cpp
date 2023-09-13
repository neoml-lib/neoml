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

#include <algorithm>

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <MemoryHandleInternal.h>
#include <MathEngineDnnConv.h>
#include <CpuMathEngineDnnChannelwiseConv.h>
#include <Rowwise/CpuRowwiseCommon.h>

namespace NeoML {

void CCpuMathEngine::BlobChannelwiseConvolution( const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	CCpuExecutionScope scope;
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );

	const float* source = GetRaw( sourceData );
	const float* filter = GetRaw( filterData );
	const float* freeTerm = ( freeTermData != nullptr ) ? GetRaw( *freeTermData ) : nullptr;
	float* result = GetRaw( resultData );

	TChannelwiseProcessFunction processFunc = GetChannelwiseProcessFunction( desc );

	const CBlobDesc& sourceDesc = desc.Source;
	const CBlobDesc& resultDesc = desc.Result;

	const int channels = sourceDesc.Channels() * sourceDesc.Depth();
	const int inputRowSize = sourceDesc.Width() * channels;
	const int outputRowSize = resultDesc.Width() * channels;

	const int inputObjectSize = inputRowSize * sourceDesc.Height();
	const int outputObjectSize = outputRowSize * resultDesc.Height();

	const int batchCount = sourceDesc.ObjectCount();
	const int resultCount = resultDesc.Height();

	const float* const sourceEnd = source + batchCount * inputObjectSize;
	for( ; source < sourceEnd; source += inputObjectSize, result += outputObjectSize ) {
		processFunc( desc, resultCount, source, /*sourceRowIndex*/0, filter, freeTerm, result, /*resultRowIndex*/0 );
	}
}

//---------------------------------------------------------------------------------------------------------------------

void CCpuMathEngine::multiplyMatrixByTransposedWithFreeTerm( const float* first, int firstHeight,
	int firstWidth, const float* second, int secondHeight, const float* freeTerm, float* result,
	const CSmallMatricesMultiplyDesc* desc )
{
	multiplyMatrixByTransposedMatrix( first, firstHeight, firstWidth, firstWidth, second,
		secondHeight, firstWidth, result, secondHeight, desc );
	if( freeTerm != nullptr ) {
		addVectorToMatrixRows( result, result, firstHeight, secondHeight, secondHeight,
			secondHeight, freeTerm );
	}
}

void CCpuMathEngine::MobileNetV3PreSEBlock( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
	const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle& expandFilterData, const CConstFloatHandle* expandFreeTermData,
	TActivationFunction expandActivation, float expandReluParam, const CConstFloatHandle& channelwiseFilterData,
	const CConstFloatHandle* channelwiseFreeTermData, TActivationFunction channelwiseActivation,
	float channelwiseReluParam, const CFloatHandle& outputHandle, const CSmallMatricesMultiplyDescsArray* SMMDescs )
{
	CCpuExecutionScope scope;
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );

	const int inputChannels = inputDesc.Channels();
	const int outputChannels = outputDesc.Channels();
	const int inputHeight = desc.Source.Height();
	const int inputWidth = desc.Source.Width();
	const int chInputRowSize = outputChannels * inputWidth;
	const int inputRowSize = inputChannels * inputWidth;
	const int outputHeight = desc.Result.Height();
	const int outputRowSize = outputChannels * desc.Result.Width();
	const int padding = desc.PaddingHeight;
	const int filterSize = desc.Filter.Width();
	const int stride = desc.StrideHeight;

	const float* inputObject = GetRaw( inputHandle );
	const float* expandFilter = GetRaw( expandFilterData );
	const float* expandFreeTerm = expandFreeTermData == nullptr ? nullptr : GetRaw( *expandFreeTermData );

	TChannelwiseProcessFunction channelwise = filterSize == 3 ? ProcessChannelwise3x3
		: ( stride == 1 ? ProcessChannelwise5x5Stride1 : ProcessChannelwise5x5Stride2 );
	const float* channelwiseFilter = GetRaw( channelwiseFilterData );
	const float* channelwiseFreeTerm = channelwiseFreeTermData == nullptr ? nullptr : GetRaw( *channelwiseFreeTermData );

	const int maxInputRowsPerStep = std::max<int>( { 1,
		( RowwiseCacheSize / ( std::max<int>( inputChannels, outputChannels ) * inputWidth ) ),
		( RowwiseMatMulRequiredHeight + inputWidth - 1 ) / inputWidth } );
	const int maxOutputRowsPerStep = std::max<int>( 1, ( RowwiseCacheSize / ( outputChannels * desc.Result.Width() ) ) );

	// Buffer for the input rows of channelwise convolution
	CFloatHandleStackVar chInputBuffVar( *this,
		std::min<int>( inputHeight, maxInputRowsPerStep + filterSize - 1 ) * chInputRowSize );

	float* chInputBuff = GetRaw( chInputBuffVar.GetHandle() );
	float* outputObject = GetRaw( outputHandle );

	for( int b = 0; b < inputDesc.ObjectCount(); ++b ) {
		const float* input = inputObject;
		float* output = outputObject;

		int inputRowsProcessed = 0;
		int outputRowsProcessed = 0;
		// The channelwise input row buffer can't hold the full image
		// That's why the buffer on each step contains [firstInputRowInBuffer, inputRowsProcessed) rows
		int firstInputRowInBuffer = 0;

		while( inputRowsProcessed < inputHeight ) {
			// Process a bunch of rows of input image (till channelwise convolution: expandConv + expandReLU)
			const int inputRowsThisStep = std::min<int>( maxInputRowsPerStep, inputHeight - inputRowsProcessed );
			float* chInput = chInputBuff + ( inputRowsProcessed - firstInputRowInBuffer ) * chInputRowSize;

			const int firstHeight = inputRowsThisStep * inputWidth;
			const int firstWidth = inputChannels;
			const int secondHeight = outputChannels;
			const CSmallMatricesMultiplyDesc* mulDesc = ( SMMDescs == nullptr ) ? nullptr :
				static_cast<const CCpuSmallMatricesMultiplyDescsArray<>*>( SMMDescs )->Get( firstHeight,
					firstHeight, firstWidth, /*secondWidth*/firstWidth, /*resultWidth*/secondHeight );

			// Apply expand convolution
			multiplyMatrixByTransposedWithFreeTerm( input, firstHeight, firstWidth,
				expandFilter, secondHeight, expandFreeTerm, chInput, mulDesc );
			MOBILENET_ACTIVATION( expandActivation, expandReluParam, chInput, inputRowsThisStep * chInputRowSize );
			inputRowsProcessed += inputRowsThisStep;

			// Calculate how many output rows we can calculate with the processed input rows
			const int outputRowsCanBeProcesed = inputRowsProcessed == inputHeight ? outputHeight
				: ( inputRowsProcessed < ( filterSize - padding ) ? 0 : ( inputRowsProcessed - filterSize + padding ) / stride + 1 );

			while( outputRowsProcessed < outputRowsCanBeProcesed ) {
				// Process channelwise output rows (while there are any)
				const int outputRowsThisStep = std::min<int>( maxOutputRowsPerStep,
					outputRowsCanBeProcesed - outputRowsProcessed );

				// Channelwise conv
				channelwise( desc, outputRowsThisStep, chInputBuff, firstInputRowInBuffer,
					channelwiseFilter, channelwiseFreeTerm, output, outputRowsProcessed );
				MOBILENET_ACTIVATION( channelwiseActivation, channelwiseReluParam, output, outputRowsThisStep * outputRowSize );

				output += outputRowsThisStep * outputRowSize;
				outputRowsProcessed += outputRowsThisStep;
			}

			input += inputRowsThisStep * inputRowSize;

			if( outputRowsProcessed < outputHeight ) {
				const int firstInputRowNeeded = outputRowsProcessed * stride - padding;
				if( firstInputRowNeeded > firstInputRowInBuffer ) {
					// Buffer for channelwise input contains rows that won't be used in future
					const int rowsToDelete = firstInputRowNeeded - firstInputRowInBuffer;
					const int rowsToMove = inputRowsProcessed - firstInputRowNeeded;
					if( rowsToMove > 0 ) {
						// There are rows which should be saved between iteration over input
						// Move them to the beginning of the buffer
						dataCopy( chInputBuff, chInputBuff + rowsToDelete * chInputRowSize,
							rowsToMove * chInputRowSize );
					}
					// Mark that channelwise input buffer now starts with a new row
					firstInputRowInBuffer = firstInputRowNeeded;
				}
			}
		}

		inputObject += inputDesc.ObjectSize();
		outputObject += outputDesc.ObjectSize();
	}
}

void CCpuMathEngine::MobileNetV3PostSEBlock( const CBlobDesc& channelwiseOutputDesc, int outputChannels,
	const CConstFloatHandle& channelwiseOutputHandle, const CConstFloatHandle& squeezeAndExciteHandle,
	const CConstFloatHandle* residualHandle, TActivationFunction activation, float reluParam,
	const CConstFloatHandle& downFilterHandle, const CConstFloatHandle* downFreeTermHandle,
	const CFloatHandle& outputHandle, const CSmallMatricesMultiplyDescsArray* SMMDescs )
{
	CCpuExecutionScope scope;
	const int inputChannels = channelwiseOutputDesc.Channels();
	const int width = channelwiseOutputDesc.Width();
	const int rowCount = channelwiseOutputDesc.Height();
	const int inputRowSize = inputChannels * width;
	const int outputRowSize = outputChannels * width;
	const int outputObjectSize = outputRowSize * rowCount;

	const int maxRowsPerStep = std::max( { 1,
		RowwiseCacheSize / ( std::max( inputChannels, outputChannels ) * width ),
		( ( RowwiseMatMulRequiredHeight + width - 1 ) / width ) } );

	CFloatHandleStackVar buffVar( *this, std::min( rowCount, maxRowsPerStep ) * inputRowSize );
	const float* inputObject = GetRaw( channelwiseOutputHandle );
	const float* squeezeVector = GetRaw( squeezeAndExciteHandle );
	const float* residualObject = residualHandle != nullptr ? GetRaw( *residualHandle ) : nullptr;
	const float* downFilter = GetRaw( downFilterHandle );
	const float* downFreeTerm = downFreeTermHandle != nullptr ? GetRaw( *downFreeTermHandle ) : nullptr;
	float* squeezed = GetRaw( buffVar.GetHandle() );
	float* outputObject = GetRaw( outputHandle );

	for( int b = 0; b < channelwiseOutputDesc.ObjectCount(); ++b ) {
		int rowsProcessed = 0;
		const float* input = inputObject;
		float* output = outputObject;
		const float* residual = residualObject;
		while( rowsProcessed < rowCount ) {
			const int rowsThisStep = std::min( rowCount - rowsProcessed, maxRowsPerStep );
			// Multiply by vector from Squeeze-and-Excite
			multiplyMatrixByDiagMatrix( input, rowsThisStep * width, inputChannels,
				squeezeVector, squeezed );
			// Activation (if present, not present means trivial linear)
			MOBILENET_ACTIVATION( activation, reluParam, squeezed, rowsThisStep * inputRowSize );

			const int firstHeight = rowsThisStep * width;
			const int firstWidth = inputChannels;
			const int secondHeight = outputChannels;
			const CSmallMatricesMultiplyDesc* mulDesc = ( SMMDescs == nullptr ) ? nullptr :
				static_cast<const CCpuSmallMatricesMultiplyDescsArray<>*>( SMMDescs )->Get( firstHeight,
					firstHeight, firstWidth, /*secondWidth*/firstWidth, /*resultWidth*/secondHeight );

			// Down-convolution (1x1)
			multiplyMatrixByTransposedWithFreeTerm( squeezed, firstHeight, firstWidth,
				downFilter, secondHeight, downFreeTerm, output, mulDesc );
			// Residual connection (if present)
			if( residual != nullptr ) {
				vectorAdd( output, residual, output, rowsThisStep * width * outputChannels );
				residual += rowsThisStep * outputRowSize;
			}

			rowsProcessed += rowsThisStep;
			input += rowsThisStep * inputRowSize;
			output += rowsThisStep * outputRowSize;
		}

		inputObject += inputRowSize * rowCount;
		squeezeVector += inputChannels;
		if( residualObject != nullptr ) {
			residualObject += outputObjectSize;
		}
		outputObject += outputObjectSize;
	}
}

CSmallMatricesMultiplyDescsArray* CCpuMathEngine::InitSmallMatricesMultiplyDescsArray()
{
	return new CCpuSmallMatricesMultiplyDescsArray</*Height*/>( *this );
}

} // namespace NeoML
