/* Copyright © 2023 ABBYY

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

#ifdef NEOML_USE_CUDA

#include <memory>
#include <algorithm>

#include "Rowwise/CudaRowwiseInterface.h"

#include "Rowwise/CudaRowwiseActivation.h"
#include "Rowwise/CudaRowwiseChConv.h"
#include "Rowwise/CudaRowwiseChConvWith1x1.h"
#include "Rowwise/CudaRowwiseConv.h"
#include "Rowwise/CudaRowwiseMobileNetV2.h"
#include "Rowwise/CudaRowwisePooling.h"
#include "Rowwise/CudaRowwiseResizeImage.h"

namespace NeoML {

CBlobDesc CCudaMathEngine::RowwiseReshape( CRowwiseOperationDesc** operations, int operationCount,
	const CBlobDesc& input )
{
	CBlobDesc output = input;
	for( int i = 0; i < operationCount; ++i ) {
		output = dynamic_cast<ICudaRowwiseImpl*>( *operations )->Reshape( output );
		++operations;
	}
	return output;
}

void CCudaMathEngine::RowwiseExecute( const CBlobDesc&, CRowwiseOperationDesc** operationDescs,
	int operationCount, const CFloatHandle& input, const CFloatHandle& output )
{
	std::vector<std::vector<ICudaRowwiseImpl*>> operations;
	for( int i = 0; i < operationCount; ++i ) {
		ICudaRowwiseImpl* operation = dynamic_cast<ICudaRowwiseImpl*>( *( operationDescs + i ) );
		if( i == 0 || !operation->IsInPlace() ) {
			operations.emplace_back();
		}
		operations.back().push_back( operation );
	}

	std::unique_ptr<CFloatHandleVar> inputBuff;
	std::unique_ptr<CFloatHandleVar> outputBuff;
	CFloatHandle currInput = input;
	CFloatHandle currOutput = output;

	for( size_t i = 0; i < operations.size(); ++i ) {
		if( i != operations.size() - 1 ) {
			if( outputBuff == nullptr || outputBuff->Size() < operations[i][0]->OutputSize() ) {
				outputBuff.reset( new CFloatHandleVar( *this, static_cast<size_t>( operations[i][0]->OutputSize() ) ) );
			}
			currOutput = outputBuff->GetHandle();
		} else {
			currOutput = output;
		}

		operations[i][0]->Process( currInput, currOutput );
		for( size_t j = 1; j < operations[i].size(); ++j ) {
			operations[i][j]->Process( currOutput, currOutput );
		}

		if( i != operations.size() - 1 ) {
			std::swap( inputBuff, outputBuff );
			currInput = currOutput;
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------

CRowwiseOperationDesc* CCudaMathEngine::InitRowwiseActivation( const CActivationDesc& desc )
{
	return new CCudaRowwiseActivation( desc );
}

CRowwiseOperationDesc* CCudaMathEngine::InitRowwiseChWith1x1( int stride, const CConstFloatHandle& channelwiseFilter,
	const CConstFloatHandle* channelwiseFreeTerm, TActivationFunction activation, float reluParam,
	const CConstFloatHandle& convFilter, const CConstFloatHandle* convFreeTerm,
	int outputChannels, bool residual )
{
	return new CCudaRowwiseChConvWith1x1( stride, channelwiseFilter, channelwiseFreeTerm, activation, reluParam,
		convFilter, convFreeTerm, outputChannels, residual );
}

CRowwiseOperationDesc* CCudaMathEngine::InitRowwiseConv( int paddingHeight, int paddingWidth, int strideHeight,
	int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filterDesc,
	const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm )
{
	return new CCudaRowwiseConv( paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight,
		dilationWidth, filterDesc, filter, freeTerm );
}

CRowwiseOperationDesc* CCudaMathEngine::InitRowwiseChConv( int paddingHeight, int paddingWidth, int strideHeight,
	int strideWidth, const CBlobDesc& filterDesc, const CConstFloatHandle& filter,
	const CConstFloatHandle* freeTerm )
{
	return new CCudaRowwiseChConv( paddingHeight, paddingWidth, strideHeight, strideWidth, filterDesc,
		filter, freeTerm );
}

CRowwiseOperationDesc* CCudaMathEngine::InitRowwiseResizeImage( TBlobResizePadding padding, float defaultValue,
	int deltaLeft, int deltaRight, int deltaTop, int deltaBottom )
{
	return new CCudaRowwiseImageResize( padding, defaultValue, deltaLeft, deltaRight, deltaTop, deltaBottom );
}

CRowwiseOperationDesc* CCudaMathEngine::InitRowwiseMobileNetV2( int, const CConstFloatHandle& expandFilter,
	const CConstFloatHandle* expandFreeTerm, int expandedChannels, TActivationFunction expandActivation,
	float expandReluParam, const CConstFloatHandle& channelwiseFilter, const CConstFloatHandle* channelwiseFreeTerm,
	int stride, TActivationFunction channelwiseActivation, float channelwiseReluParam,
	const CConstFloatHandle& downFilter, const CConstFloatHandle* downFreeTerm, int outputChannels, bool residual )
{
	return new CCudaRowwiseMobileNetV2( expandFilter, expandFreeTerm, expandedChannels, expandActivation, expandReluParam,
		channelwiseFilter, channelwiseFreeTerm, stride, channelwiseActivation, channelwiseReluParam, downFilter,
		downFreeTerm, outputChannels, residual );
}

CRowwiseOperationDesc* CCudaMathEngine::InitRowwise2DPooling( bool isMax, int filterHeight, int filterWidth,
	int strideHeight, int strideWidth )
{
	return new CCudaRowwise2DPooling( *this, isMax, filterHeight, filterWidth, strideHeight, strideWidth );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
