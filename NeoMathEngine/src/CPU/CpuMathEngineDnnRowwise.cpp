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

#include <memory>
#include <vector>
#include <algorithm>

#include <CpuMathEngine.h>
#include <CpuMathEnginePrivate.h>
#include <CpuExecutionScope.h>
#include <MemoryHandleInternal.h>

#include <Rowwise/CpuRowwiseActivation.h>
#include <Rowwise/CpuRowwiseBuffer.h>
#include <Rowwise/CpuRowwiseChConv.h>
#include <Rowwise/CpuRowwiseChConvWith1x1.h>
#include <Rowwise/CpuRowwiseConv.h>
#include <Rowwise/CpuRowwiseMobileNetV2.h>
#include <Rowwise/CpuRowwisePooling.h>
#include <Rowwise/CpuRowwiseResizeImage.h>

namespace NeoML {

CBlobDesc CCpuMathEngine::RowwiseReshape( CRowwiseOperationDesc** operations, int operationCount,
	const CBlobDesc& input )
{
	CBlobDesc output = input;
	for( int i = 0; i < operationCount; ++i ) {
		output = dynamic_cast<ICpuRowwiseImpl*>( *operations )->Reshape( output );
		++operations;
	}
	return output;
}

static constexpr int RowwiseMaxBuffSize = 32 * 1024;

typedef std::vector<std::vector<ICpuRowwiseImpl*>> CRowwiseSubchains;
typedef std::vector<std::unique_ptr<ICpuRowwiseBuffer>> CRowwiseBuffers;

// Splits chain of rowwise operation into subchains
// First operation in each subchain is performed out-of-place
// the rest of operations in each subchain are trivial and performed in-place
// most common example:
//     conv -> relu6 -> conv -> relu6 -> pool
// will be split into:
//     [conv -> relu6] -> [conv -> relu6] -> [pool]
// Returns the buffer size needed for calculation (size in floats)
static int splitIntoSubchains( CRowwiseOperationDesc** operationDescs, int operationCount,
	CRowwiseSubchains& subchains )
{
	int inOperationBufferSize = 0;

	for( int i = 0; i < operationCount; ++i ) {
		ICpuRowwiseImpl* operation = dynamic_cast<ICpuRowwiseImpl*>( *( operationDescs + i ) );
		if( i == 0 || !operation->IsTrivial() ) {
			subchains.emplace_back();
		}
		inOperationBufferSize = std::max( inOperationBufferSize, operation->InOperationBufferSize() );
		subchains.back().push_back( operation );
	}

	return inOperationBufferSize;
}

// Allocates rowwise buffers for the current set of operations
static void allocateRowwiseBuffers( const CBlobDesc& inputDesc, const CRowwiseSubchains& operations,
	const CFloatHandle& input, const CFloatHandle& output, CRowwiseBuffers& buffers )
{
	buffers.reserve( operations.size() + 1 );

	const int inputRowsCount = inputDesc.ObjectCount() * inputDesc.Height();
	buffers.emplace_back( new CCpuRowwiseWrapper( GetRaw( input ), inputRowsCount,
		inputDesc.Width() * inputDesc.Channels() ) );
	// Each time we try to allocate buffer which meets the output requirement of latest operator
	// and the input requirement of next operator (skipping operators which don't have such requirements)
	int prevOutputRequirement = 1;
	int nextInputRequirement = 1;
	size_t nextInputRequirementIndex = 0;
	for( size_t i = 0; i < operations.size() - 1; ++i ) {
		if( i == nextInputRequirementIndex ) {
			// Finding the next operation which has requirement for its input
			nextInputRequirement = 1;
			while( nextInputRequirementIndex < operations.size() ) {
				++nextInputRequirementIndex;
				if( nextInputRequirementIndex < operations.size()
					&& operations[nextInputRequirementIndex][0]->InputRowRequirement() > 0 )
				{
					nextInputRequirement = operations[nextInputRequirementIndex][0]->InputRowRequirement();
					break;
				}
			}
		}

		if( operations[i][0]->OutputRowRequirement() > 0 ) {
			prevOutputRequirement = operations[i][0]->OutputRowRequirement();
		}

		const int rowSize = operations[i][0]->OutputRowSize();
		const int maxRowCount = std::min( operations[i][0]->OutputRowCount(),
			std::max( { prevOutputRequirement, nextInputRequirement, RowwiseMaxBuffSize / rowSize } ) );
		buffers.emplace_back( new CCpuRowwiseBuffer( *input.GetMathEngine(),
			maxRowCount, rowSize, operations[i][0]->OutputRowCount() ) );
	}
	buffers.emplace_back( new CCpuRowwiseWrapper( GetRaw( output ), operations.back().back()->OutputRowCount(),
		operations.back().back()->OutputRowSize() ) );
}

// Processes corner case: single subchain which has multiple operations
void executeSingleSubchain( std::vector<ICpuRowwiseImpl*>& subchain, CRowwiseBuffers& buffers, float* inOperationBuffer )
{
	const int maxOutputRowsPerStep = std::max( 1, RowwiseMaxBuffSize / subchain.back()->OutputRowSize() );
	while( buffers.back()->EmptyRowCount() > 0 ) {
		// In this case all the data is allocated (cause input and output blobs are fully allocated by CDnn)
		// But because of the fact that we have multiple operations in subchain it will be ineffective to
		// call Process for the whole data for each of operations (because data may be too big)
		// That's why we manually limit the number of output rows calculated during each call
		const int outputRowsThisStep = std::min( maxOutputRowsPerStep, buffers.back()->EmptyRowCount() );
		ICpuRowwiseImpl::CProcessingReport report = subchain[0]->Process( buffers[0]->DataRows(),
			buffers[0]->DataRowIndex(), buffers[0]->DataRowCount(), buffers[1]->EmptyRows(),
			buffers[1]->DataRowProcessed(), outputRowsThisStep, inOperationBuffer );
		PRESUME_EXPR( report.OutputRowsCalculated == outputRowsThisStep );

		for( size_t j = 1; j < subchain.size(); ++j ) {
			( void ) subchain[j]->Process( buffers[1]->EmptyRows(), buffers[1]->DataRowProcessed(),
				report.OutputRowsCalculated, buffers[1]->EmptyRows(), buffers[1]->DataRowProcessed(),
				report.OutputRowsCalculated, inOperationBuffer );
		}

		buffers[1]->AddRows( report.OutputRowsCalculated );
		if( report.InputRowsMayBeRemoved > 0 ) {
			buffers[0]->RemoveRows( report.InputRowsMayBeRemoved );
		}
	}
	return;
}

void CCpuMathEngine::RowwiseExecute( const CBlobDesc& inputDesc, CRowwiseOperationDesc** operationDescs,
	int operationCount, const CFloatHandle& input, const CFloatHandle& output )
{
	PRESUME_EXPR( operationCount > 1 );
	PRESUME_EXPR( inputDesc.Depth() == 1 );

	CCpuExecutionScope scope;

	CRowwiseSubchains operations;
	int inOperationBufferSize = splitIntoSubchains( operationDescs, operationCount, operations );

	CRowwiseBuffers buffers;
	allocateRowwiseBuffers( inputDesc, operations, input, output, buffers );

	std::unique_ptr<CFloatHandleStackVar> inOperationBufferVar;
	float* inOperationBuffer = nullptr;
	if( inOperationBufferSize > 0 ) {
		inOperationBufferVar.reset( new CFloatHandleStackVar( *this, static_cast<size_t>( inOperationBufferSize ) ) );
		inOperationBuffer = GetRaw( inOperationBufferVar->GetHandle() );
	}

	buffers.front()->AddRows( inputDesc.ObjectCount() * inputDesc.Height() );

	if( operations.size() == 1 && operations[0].size() > 1 ) {
		executeSingleSubchain( operations[0], buffers, inOperationBuffer );
		return;
	}

	while( buffers.back()->EmptyRowCount() > 0 ) {
		for( size_t i = 0; i < operations.size(); ++i ) {
			if( buffers[i]->DataRowCount() == 0 || buffers[i + 1]->EmptyRowCount() == 0 ) {
				continue;
			}

			ICpuRowwiseImpl::CProcessingReport report = operations[i][0]->Process( buffers[i]->DataRows(),
				buffers[i]->DataRowIndex(), buffers[i]->DataRowCount(), buffers[i + 1]->EmptyRows(),
				buffers[i + 1]->DataRowProcessed(), buffers[i + 1]->EmptyRowCount(), inOperationBuffer );

			for( size_t j = 1; j < operations[i].size(); ++j ) {
				( void ) operations[i][j]->Process( buffers[i + 1]->EmptyRows(), buffers[i + 1]->DataRowProcessed(),
					report.OutputRowsCalculated, buffers[i + 1]->EmptyRows(), buffers[i + 1]->DataRowProcessed(),
					report.OutputRowsCalculated, inOperationBuffer );
			}

			if( report.OutputRowsCalculated > 0 ) {
				buffers[i + 1]->AddRows( report.OutputRowsCalculated );
			}

			if( report.InputRowsMayBeRemoved > 0 ) {
				buffers[i]->RemoveRows( report.InputRowsMayBeRemoved );
			}

			// Try to fill as much of output as possible
			// Significanlty reduces a number of calls with report.OutputRowsCalculated == 0
			if( buffers[i + 1]->EmptyRowCount() > 0
				&& ( ( i == 0 && buffers[i]->DataRowCount() > 0 )
					|| ( i > 0 && buffers[i]->DataRowProcessed() < operations[i - 1][0]->OutputRowCount() ) ) )
			{
				break;
			}
		}
	}
}

//------------------------------------------------------------------------------------------------------------

CRowwiseOperationDesc* CCpuMathEngine::InitRowwiseActivation( const CActivationDesc& desc )
{
	return new CCpuRowwiseActivation( desc );
}

//------------------------------------------------------------------------------------------------------------

CRowwiseOperationDesc* CCpuMathEngine::InitRowwiseChConv( int paddingHeight, int paddingWidth, int strideHeight,
	int strideWidth, const CBlobDesc& filterDesc, const CConstFloatHandle& filter,
	const CConstFloatHandle* freeTerm )
{
	return new CCpuRowwiseChConv( paddingHeight, paddingWidth, strideHeight, strideWidth, filterDesc, GetRaw( filter ),
		freeTerm == nullptr ? nullptr : GetRaw( *freeTerm ) );
}

//------------------------------------------------------------------------------------------------------------

CRowwiseOperationDesc* CCpuMathEngine::InitRowwiseConv( int paddingHeight, int paddingWidth, int strideHeight,
	int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filterDesc,
	const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm )
{
	return new CCpuRowwiseConv( *this, filterDesc.Channels(), paddingHeight, paddingWidth, strideHeight, strideWidth,
		dilationHeight, dilationWidth, filterDesc.ObjectCount(), filterDesc.Height(), filterDesc.Width(),
		GetRaw( filter ), freeTerm == nullptr ? nullptr : GetRaw( *freeTerm ) );
}

//------------------------------------------------------------------------------------------------------------

void CCpuMathEngine::ChannelwiseWith1x1( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
	const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle& channelwiseFilterData, const CConstFloatHandle* channelwiseFreeTermData,
	TActivationFunction activation, float reluParam, const CConstFloatHandle& convFilterData,
	const CConstFloatHandle* convFreeTermData, bool residual, const CFloatHandle& outputHandle )
{
	CCpuExecutionScope scope;
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );

	CCpuRowwiseChConvWith1x1 impl( *this, desc.StrideHeight, GetRaw( channelwiseFilterData ),
		channelwiseFreeTermData == nullptr ? nullptr : GetRaw( *channelwiseFreeTermData ),
		activation, reluParam, GetRaw( convFilterData ),
		convFreeTermData == nullptr ? nullptr : GetRaw( *convFreeTermData ),
		outputDesc.Channels(), residual );
	(void)impl.Reshape( inputDesc );

	// Buffer for the output rows of channelwise convolution
	CFloatHandleStackVar bufferVar( *this, static_cast<size_t>( impl.InOperationBufferSize() ) );

	float* buffer = GetRaw( bufferVar.GetHandle() );
	const float* input = GetRaw( inputHandle );
	float* output = GetRaw( outputHandle );

	const int inputRowCount = desc.Source.ObjectCount() * desc.Source.Height();
	const int outputRowCount = desc.Result.ObjectCount() * desc.Result.Height();
	ICpuRowwiseImpl::CProcessingReport report = impl.Process( input, 0, inputRowCount,
		output, 0, outputRowCount, buffer );
	( void ) report; // Avoid compiler warning in release configuration
	PRESUME_EXPR( report.InputRowsMayBeRemoved == inputRowCount );
	PRESUME_EXPR( report.OutputRowsCalculated == outputRowCount );
}

CRowwiseOperationDesc* CCpuMathEngine::InitRowwiseChWith1x1( int stride, const CConstFloatHandle& channelwiseFilter,
	const CConstFloatHandle* channelwiseFreeTerm, TActivationFunction activation, float reluParam,
	const CConstFloatHandle& convFilter, const CConstFloatHandle* convFreeTerm, int outputChannels, bool residual )
{
	return new CCpuRowwiseChConvWith1x1( *this, stride, GetRaw( channelwiseFilter ),
		channelwiseFreeTerm == nullptr ? nullptr : GetRaw( *channelwiseFreeTerm ),
		activation, reluParam, GetRaw( convFilter ),
		convFreeTerm == nullptr ? nullptr : GetRaw( *convFreeTerm ),
		outputChannels, residual );
}

//------------------------------------------------------------------------------------------------------------

void CCpuMathEngine::MobileNetV2Block( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
	const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle& expandFilterData, const CConstFloatHandle* expandFreeTermData,
	TActivationFunction expandActivation, float expandReluParam, const CConstFloatHandle& channelwiseFilterData,
	const CConstFloatHandle* channelwiseFreeTermData, TActivationFunction channelwiseActivation,
	float channelwiseReluParam, const CConstFloatHandle& downFilterData, const CConstFloatHandle* downFreeTermData,
	bool residual, const CFloatHandle& outputHandle )
{
	CCpuExecutionScope scope;
	const CCommonChannelwiseConvolutionDesc& desc = static_cast< const CCommonChannelwiseConvolutionDesc& >( convDesc );

	CCpuRowwiseMobileNetV2 blockImpl( *this, inputDesc.Channels(), GetRaw( expandFilterData ),
		expandFreeTermData == nullptr ? nullptr : GetRaw( *expandFreeTermData ),
		desc.Source.Channels(), expandActivation, expandReluParam, GetRaw( channelwiseFilterData ),
		channelwiseFreeTermData == nullptr ? nullptr : GetRaw( *channelwiseFreeTermData ),
		desc.StrideHeight, channelwiseActivation, channelwiseReluParam, GetRaw( downFilterData ),
		downFreeTermData == nullptr ? nullptr : GetRaw( *downFreeTermData ),
		outputDesc.Channels(), residual );
	const CBlobDesc reshapeResult = blockImpl.Reshape( inputDesc );
	( void ) reshapeResult;
	PRESUME_EXPR( reshapeResult.HasEqualDimensions( outputDesc ) );
	CFloatHandleStackVar buffer( *this, blockImpl.InOperationBufferSize() );
	const ICpuRowwiseImpl::CProcessingReport report = blockImpl.Process( GetRaw( inputHandle ), 0,
		inputDesc.ObjectCount() * inputDesc.Height(), GetRaw( outputHandle ), 0,
		outputDesc.ObjectCount() * outputDesc.Height(), GetRaw( buffer.GetHandle() ) );
	( void ) report;
	PRESUME_EXPR( report.InputRowsMayBeRemoved == inputDesc.ObjectCount() * inputDesc.Height() );
	PRESUME_EXPR( report.OutputRowsCalculated == outputDesc.ObjectCount() * outputDesc.Height() );
}

CRowwiseOperationDesc* CCpuMathEngine::InitRowwiseMobileNetV2( int inputChannels,
	const CConstFloatHandle& expandFilter, const CConstFloatHandle* expandFreeTerm, int expandedChannels,
	TActivationFunction expandActivation, float expandReluParam,
	const CConstFloatHandle& channelwiseFilter, const CConstFloatHandle* channelwiseFreeTerm, int stride,
	TActivationFunction channelwiseActivation, float channelwiseReluParam,
	const CConstFloatHandle& downFilter, const CConstFloatHandle* downFreeTerm, int outputChannels, bool residual )
{
	return new CCpuRowwiseMobileNetV2( *this, inputChannels, GetRaw( expandFilter ),
		expandFreeTerm == nullptr ? nullptr : GetRaw( *expandFreeTerm ),
		expandedChannels, expandActivation, expandReluParam, GetRaw( channelwiseFilter ),
		channelwiseFreeTerm == nullptr ? nullptr : GetRaw( *channelwiseFreeTerm ),
		stride, channelwiseActivation, channelwiseReluParam, GetRaw( downFilter ),
		downFreeTerm == nullptr ? nullptr : GetRaw( *downFreeTerm ),
		outputChannels, residual );
}

//---------------------------------------------------------------------------------------------------

CRowwiseOperationDesc* CCpuMathEngine::InitRowwise2DPooling( bool isMax, int filterHeight, int filterWidth,
	int strideHeight, int strideWidth )
{
	return new CCpuRowwise2DPooling( *this, isMax, filterHeight, filterWidth, strideHeight, strideWidth );
}

//------------------------------------------------------------------------------------------------------------

void CCpuMathEngine::BlobResizeImage( const CBlobDesc& from, const CFloatHandle& fromData, int deltaLeft, int deltaRight,
	int deltaTop, int deltaBottom, TBlobResizePadding padding, float defaultValue,
	const CBlobDesc& to, const CFloatHandle& toData )
{
	CCpuExecutionScope scope;

	CCpuRowwiseImageResize impl( padding, defaultValue, deltaLeft, deltaRight, deltaTop, deltaBottom );
	CBlobDesc reshapeResult = impl.Reshape( from );
	( void ) reshapeResult;
	PRESUME_EXPR( reshapeResult.HasEqualDimensions( to ) );

	const float* fromPtr = GetRaw( fromData );
	float* toPtr = GetRaw( toData );

	const int currThreadCount = IsOmpRelevant( from.ObjectCount(), from.BlobSize() ) ? threadCount : 1;
	NEOML_OMP_FOR_NUM_THREADS( currThreadCount )
		for( int batch = 0; batch < from.ObjectCount(); ++batch ) {
			ICpuRowwiseImpl::CProcessingReport report = impl.Process( fromPtr + batch * from.ObjectSize(), batch * from.Height(),
				from.Height(), toPtr + batch * to.ObjectSize(), batch * to.Height(), to.Height(), nullptr );
			( void ) report;
			PRESUME_EXPR( report.OutputRowsCalculated == to.Height() );
		}
}

CRowwiseOperationDesc* CCpuMathEngine::InitRowwiseResizeImage( TBlobResizePadding padding, float defaultValue,
	int deltaLeft, int deltaRight, int deltaTop, int deltaBottom )
{
	return new CCpuRowwiseImageResize( padding, defaultValue, deltaLeft, deltaRight, deltaTop, deltaBottom );
}

} // namespace NeoML
