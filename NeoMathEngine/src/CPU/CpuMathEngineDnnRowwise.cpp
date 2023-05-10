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
#include <CpuMathEngineDnnRowwise.h>
#include <CpuMathEnginePrivate.h>
#include <CpuExecutionScope.h>
#include <MemoryHandleInternal.h>

namespace NeoML {

CRowwiseWrapper::CRowwiseWrapper( float* data, int rowCount, int rowSize ) :
	firstDataRow( data ),
	rowCount( rowCount ),
	rowSize( rowSize ),
	addedRows( 0 ),
	removedRows( 0 )
{}

int CRowwiseWrapper::DataRowsCount() const
{
	PRESUME_EXPR( addedRows >= removedRows );
	return addedRows - removedRows;
}

const float* CRowwiseWrapper::DataRows() const
{
	PRESUME_EXPR( DataRowsCount() > 0 );
	return firstDataRow;
}

float* CRowwiseWrapper::EmptyRows()
{
	PRESUME_EXPR( addedRows < rowCount );
	return firstDataRow + DataRowsCount() * rowSize;
}

void CRowwiseWrapper::AddRows( int count )
{
	PRESUME_EXPR( count > 0 );
	addedRows += count;
	PRESUME_EXPR( addedRows <= rowCount );
}

void CRowwiseWrapper::RemoveRows( int count )
{
	PRESUME_EXPR( count > 0 );
	PRESUME_EXPR( count <= DataRowsCount() );
	removedRows += count;
	firstDataRow += count * rowSize;
}

void CRowwiseWrapper::Reset()
{
	removedRows = 0;
	addedRows = 0;
}

CRowwiseBuffer::CRowwiseBuffer( IMathEngine& mathEngine, int rowCount, int rowSize ) :
	rowCount( rowCount ),
	rowSize( rowSize ),
	bufferVar( mathEngine, rowCount * rowSize ),
	bufferPtr( GetRaw( bufferVar.GetHandle() ) ),
	dataRowsCount( 0 ),
	dataRowIndex( 0 )
{
}

const float* CRowwiseBuffer::DataRows() const
{
	PRESUME_EXPR( dataRowsCount > 0 );
	return bufferPtr;
}

float* CRowwiseBuffer::EmptyRows()
{
	PRESUME_EXPR( EmptyRowsCount() > 0 );
	return bufferPtr + dataRowsCount * rowSize;
}

void CRowwiseBuffer::AddRows( int count )
{
	PRESUME_EXPR( count > 0 );
	dataRowsCount += count;
	PRESUME_EXPR( dataRowsCount <= rowCount );
}

void CRowwiseBuffer::RemoveRows( int count )
{
	PRESUME_EXPR( count > 0 );
	PRESUME_EXPR( count <= dataRowsCount );
	dataRowsCount -= count;
	dataRowIndex += count;
	if( dataRowsCount > 0 ) {
		// Move remaining data rows to the beginning of buffer
		// HACK: we know that data copying work sequentially that's why we don't need to check for the overlap
		dataCopy( bufferPtr, bufferPtr + count * rowSize, dataRowsCount * rowSize );
	}
}

void CRowwiseBuffer::Reset()
{
	dataRowsCount = 0;
	dataRowIndex = 0;
}

//---------------------------------------------------------------------------------------------------------------------

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
	PRESUME_EXPR( operationCount > 0 );
	PRESUME_EXPR( inputDesc.Depth() == 1 );

	CCpuExecutionScope scope;

	std::vector<IRowwiseCpuImpl*> operations;
	operations.reserve( static_cast<size_t>( operationCount ) );
	for( int i = 0; i < operationCount; ++i ) {
		operations.push_back( dynamic_cast<IRowwiseCpuImpl*>( *( operationDescs + i ) ) );
	}

	std::vector<std::unique_ptr<IRowwiseBuffer>> buffers;
	buffers.reserve( static_cast<size_t>( operationCount + 1 ) );

	buffers.emplace_back( new CRowwiseWrapper( GetRaw( input ), inputDesc.Height(),
		inputDesc.Width() * inputDesc.Channels() ) );
	int inOperationBufferSize = 0;
	for( size_t i = 0; i < operations.size() - 1; ++i ) {
		inOperationBufferSize = std::max( inOperationBufferSize, operations[i]->InOperationBufferSize() );
		const int rowSize = operations[i]->OutputRowSize();
		const int maxRowCount = std::min( std::max( operations[i + 1]->RequiredRowsCount(), RowwiseMaxBuffSize / rowSize ),
			operations[i]->OutputHeight() );
		buffers.emplace_back( new CRowwiseBuffer( *this, maxRowCount, rowSize ) );
	}
	buffers.emplace_back( new CRowwiseWrapper( GetRaw( output ), operations.back()->OutputHeight(),
			operations.back()->OutputRowSize() ) );

	inOperationBufferSize = std::max( inOperationBufferSize, operations.back()->InOperationBufferSize() );
	std::unique_ptr<CFloatHandleStackVar> inOperationBuffer;
	if( inOperationBufferSize > 0 ) {
		inOperationBuffer.reset( new CFloatHandleStackVar( *this, static_cast<size_t>( inOperationBufferSize ) ) );
	}

	for( int b = 0; b < inputDesc.ObjectCount(); ++b ) {
		for( std::unique_ptr<IRowwiseBuffer>& buffer : buffers ) {
			buffer->Reset();
		}
		buffers.front()->AddRows( inputDesc.Height() );

		while( buffers.back()->EmptyRowsCount() > 0 ) {
			for( size_t i = 0; i < operations.size(); ++i ) {
				if( buffers[i]->DataRowsCount() == 0 || buffers[i + 1]->EmptyRowsCount() == 0 ) {
					continue;
				}

				IRowwiseCpuImpl::CProcessingReport report = operations[i]->Process( buffers[i]->DataRows(),
					buffers[i]->DataRowIndex(), buffers[i]->DataRowsCount(), buffers[i + 1]->EmptyRows(),
					buffers[i + 1]->DataRowIndex() + buffers[i + 1]->DataRowsCount(), buffers[i + 1]->EmptyRowsCount(),
					inOperationBuffer == nullptr ? nullptr : GetRaw( inOperationBuffer->GetHandle() ) );

				if( report.OutputRowsCalculated > 0 ) {
					buffers[i + 1]->AddRows( report.OutputRowsCalculated );
				}

				if( report.InputRowsMayBeRemoved > 0 ) {
					buffers[i]->RemoveRows( report.InputRowsMayBeRemoved );
				}

				// Try to fill as much of output as possible
				// Significanlty reduces a number of calls with report.OutputRowsCalculated == 0
				if( buffers[i + 1]->EmptyRowsCount() > 0
					&& buffers[i + 1]->DataRowsProcessed() < operations[i]->OutputHeight()
					&& ( ( i == 0 && buffers[i]->DataRowsCount() > 0 )
						|| ( i > 0 && buffers[i]->DataRowsProcessed() < operations[i - 1]->OutputHeight() ) ) )
				{
					break;
				}
			}
		}

		buffers.back()->RemoveRows( operations.back()->OutputHeight() );
	}
}

} // namespace NeoML
