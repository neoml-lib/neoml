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

#include "Rowwise/CpuRowwiseActivation.h"

namespace NeoML {

CBlobDesc CCpuMathEngine::RowwiseReshape( CRowwiseOperationDesc** operations, int operationCount,
	const CBlobDesc& input )
{
	CBlobDesc output = input;
	for( int i = 0; i < operationCount; ++i ) {
		output = dynamic_cast<IRowwiseCpuImpl*>( *operations )->Reshape( output );
		++operations;
	}
	return output;
}

static constexpr int RowwiseMaxBuffSize = 32 * 1024;

void CCpuMathEngine::RowwiseExecute( const CBlobDesc& inputDesc, CRowwiseOperationDesc** operationDescs,
	int operationCount, const CFloatHandle& input, const CFloatHandle& output )
{
	PRESUME_EXPR( operationCount > 1 );
	PRESUME_EXPR( inputDesc.Depth() == 1 );

	CCpuExecutionScope scope;

	int inOperationBufferSize = 0;
	std::vector<std::vector<IRowwiseCpuImpl*>> operations;
	for( int i = 0; i < operationCount; ++i ) {
		IRowwiseCpuImpl* operation = dynamic_cast<IRowwiseCpuImpl*>( *( operationDescs + i ) );
		if( i == 0 || !operation->IsTrivial() ) {
			operations.emplace_back();
		}
		inOperationBufferSize = std::max( inOperationBufferSize, operation->InOperationBufferSize() );
		operations.back().push_back( operation );
	}

	std::vector<std::unique_ptr<IRowwiseBuffer>> buffers;
	buffers.reserve( static_cast<size_t>( operationCount + 1 ) );

	const int inputRowsCount = inputDesc.ObjectCount() * inputDesc.Height();
	buffers.emplace_back( new CRowwiseWrapper( GetRaw( input ), inputRowsCount,
		inputDesc.Width() * inputDesc.Channels() ) );
	for( size_t i = 0; i < operations.size() - 1; ++i ) {
		const int rowSize = operations[i][0]->OutputRowSize();
		const int maxRowCount = std::min( std::max( operations[i + 1][0]->MinInputRowCount(), RowwiseMaxBuffSize / rowSize ),
			operations[i][0]->OutputRowCount() );
		buffers.emplace_back( new CRowwiseBuffer( *this, maxRowCount, rowSize, operations[i][0]->OutputRowCount() ) );
	}
	buffers.emplace_back( new CRowwiseWrapper( GetRaw( output ), operations.back().back()->OutputRowCount(),
		operations.back().back()->OutputRowSize() ) );

	std::unique_ptr<CFloatHandleStackVar> inOperationBufferVar;
	float* inOperationBuffer = nullptr;
	if( inOperationBufferSize > 0 ) {
		inOperationBufferVar.reset( new CFloatHandleStackVar( *this, static_cast<size_t>( inOperationBufferSize ) ) );
		inOperationBuffer = GetRaw( inOperationBufferVar->GetHandle() );
	}

	buffers.front()->AddRows( inputDesc.ObjectCount() * inputDesc.Height() );

	if( operations.size() == 1 && operations[0].size() > 1 ) {
		const int maxOutputRowsPerStep = std::max( 1, RowwiseMaxBuffSize / operations.back().back()->OutputRowSize() );
		while( buffers.back()->EmptyRowCount() > 0 ) {
			const int outputRowsThisStep = std::min( maxOutputRowsPerStep, buffers.back()->EmptyRowCount() );
			IRowwiseCpuImpl::CProcessingReport report = operations[0][0]->Process( buffers[0]->DataRows(),
				buffers[0]->DataRowIndex(), buffers[0]->DataRowCount(), buffers[1]->EmptyRows(),
				buffers[1]->DataRowProcessed(), outputRowsThisStep, inOperationBuffer );
			PRESUME_EXPR( report.OutputRowsCalculated == outputRowsThisStep );

			for( size_t j = 1; j < operations[0].size(); ++j ) {
				( void ) operations[0][j]->Process( buffers[1]->EmptyRows(), buffers[1]->DataRowProcessed(),
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

	while( buffers.back()->EmptyRowCount() > 0 ) {
		for( size_t i = 0; i < operations.size(); ++i ) {
			if( buffers[i]->DataRowCount() == 0 || buffers[i + 1]->EmptyRowCount() == 0 ) {
				continue;
			}

			IRowwiseCpuImpl::CProcessingReport report = operations[i][0]->Process( buffers[i]->DataRows(),
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
				&& buffers[i + 1]->DataRowProcessed() < operations[i][0]->OutputRowCount()
				&& ( ( i == 0 && buffers[i]->DataRowCount() > 0 )
					|| ( i > 0 && buffers[i]->DataRowProcessed() < operations[i - 1][0]->OutputRowCount() ) ) )
			{
				break;
			}
		}
	}
}

CRowwiseOperationDesc* CCpuMathEngine::InitRowwiseActivation( TActivationFunction activation,
	float param0, float param1 )
{
	return new CRowwiseActivation( activation, param0, param1 );
}

} // namespace NeoML
