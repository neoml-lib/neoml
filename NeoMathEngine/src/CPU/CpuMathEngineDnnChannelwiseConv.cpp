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
	const float* freeTerm = freeTermData != 0 ? GetRaw( *freeTermData ) : 0;
	float* result = GetRaw( resultData );

	TChannelwiseProcessFunction processFunc = GetChannelwiseProcessFunction( desc );

	const CBlobDesc& sourceDesc = desc.Source;
	const CBlobDesc& filterDesc = desc.Filter;
	const CBlobDesc& resultDesc = desc.Result;

	const int curThreadCount = IsOmpRelevant( sourceDesc.ObjectCount() * resultDesc.Height(),
		static_cast<int64_t>( sourceDesc.BlobSize() ) * filterDesc.BlobSize() ) ? threadCount : 1;

	const int channels = sourceDesc.Channels() * sourceDesc.Depth();

	const int inputRowSize = sourceDesc.Width() * channels;
	const int outputRowSize = resultDesc.Width() * channels;

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
			float* res = result + batchStart * outputObjectSize + resultStart * outputRowSize;

			for( ; src < srcEnd; src += inputObjectSize, res += outputObjectSize ) {
				processFunc( desc, resultCount, src, 0, filter, freeTerm, res, resultStart );
			}
		}
	}
}

//=====================================================================================================================

void CCpuMathEngine::multiplyMatrixByTransposedWithFreeTerm( const float* first, int firstHeight,
	int firstWidth, const float* second, int secondHeight, const float* freeTerm, float* result )
{
	multiplyMatrixByTransposedMatrix( first, firstHeight, firstWidth, firstWidth, second,
		secondHeight, firstWidth, result, secondHeight );
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
	float channelwiseReluParam, const CFloatHandle& outputHandle )
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

	const int maxInputRowsPerStep = std::max<int>( 1,
		( RowwiseCacheSize / ( std::max<int>( inputChannels, outputChannels ) * inputWidth ) ) );
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

			// Apply expand convolution
			multiplyMatrixByTransposedWithFreeTerm( input, inputRowsThisStep * inputWidth, inputChannels,
				expandFilter, outputChannels, expandFreeTerm, chInput );
			if( expandActivation == AF_HSwish ) {
				vectorHSwish( chInput, chInput, inputRowsThisStep * chInputRowSize );
			} else if( expandActivation == AF_ReLU ) {
				if( expandReluParam > 0 ) {
					vectorReLU( chInput, chInput, inputRowsThisStep * chInputRowSize, expandReluParam );
				} else {
					vectorReLU( chInput, chInput, inputRowsThisStep * chInputRowSize );
				}
			}
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
				if( channelwiseActivation == AF_HSwish ) {
					vectorHSwish( output, output, outputRowsThisStep * outputRowSize );
				} else if( channelwiseActivation == AF_ReLU ) {
					if( channelwiseReluParam > 0 ) {
						vectorReLU( output, output, outputRowsThisStep * outputRowSize, channelwiseReluParam );
					} else {
						vectorReLU( output, output, outputRowsThisStep * outputRowSize );
					}
				}

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
	const CFloatHandle& outputHandle )
{
	CCpuExecutionScope scope;
	const int inputChannels = channelwiseOutputDesc.Channels();
	const int width = channelwiseOutputDesc.Width();
	const int rowCount = channelwiseOutputDesc.Height();
	const int inputRowSize = inputChannels * width;
	const int outputRowSize = outputChannels * width;
	const int outputObjectSize = outputRowSize * rowCount;

	const int maxRowsPerStep = std::max( 1, ( RowwiseCacheSize / ( std::max( inputChannels, outputChannels ) * width ) ) );

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
			if( activation == AF_HSwish ) {
				vectorHSwish( squeezed, squeezed, rowsThisStep * inputRowSize );
			} else if( activation == AF_ReLU ) {
				if( reluParam > 0 ) {
					vectorReLU( squeezed, squeezed, rowsThisStep * inputRowSize, reluParam );
				} else {
					vectorReLU( squeezed, squeezed, rowsThisStep * inputRowSize );
				}
			}
			// Down-convolution (1x1)
			multiplyMatrixByTransposedWithFreeTerm( squeezed, rowsThisStep * width, inputChannels,
				downFilter, outputChannels, downFreeTerm, output );
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

} // namespace NeoML
