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

#include <NeoMathEngine/BlobDesc.h>
#include <CpuMathEnginePrivate.h>

namespace NeoML {

// CPU implementation of rowwise operation
class IRowwiseCpuImpl {
public:
	virtual ~IRowwiseCpuImpl() = default;

	virtual int MinInputRowCount() const = 0;

	virtual CBlobDesc Reshape( const CBlobDesc& inputSize ) = 0;
	virtual int InOperationBufferSize() const = 0;
	virtual int OutputRowCount() const = 0;
	virtual int OutputRowSize() const = 0;
	virtual bool IsInPlace() const = 0;

	// The result of single rowwise processing
	struct CProcessingReport {
		int OutputRowsCalculated; // number of output rows calculated during this call
		int InputRowsMayBeRemoved; // number of input rows which are not needed anymore
	};

	virtual CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const = 0;
};

//=====================================================================================================================

// Interface for buffers for rowwise operations
// First DataRowCount() rows in buffer are filled with actual data
// Rest of the rows are considered empty
class IRowwiseBuffer {
public:
	virtual ~IRowwiseBuffer() = default;

	// Number of elements in row
	virtual int RowSize() const = 0;

	// Index of first data row in buffer
	virtual int DataRowIndex() const = 0;
	// Number of rows in buffer filled with data
	virtual int DataRowCount() const = 0;
	// Number of data rows ever appeared in this buffer
	virtual int DataRowProcessed() const { return DataRowIndex() + DataRowCount(); }
	// Pointer to the beginning of rows with data
	virtual const float* DataRows() const = 0;

	// Empty rows are the rows which go immediately after the data
	// Their number depends on the size of memory allocated for this buffer
	// Number of empty rows in buffer
	virtual int EmptyRowCount() const = 0;
	// Pointer after data rows
	virtual float* EmptyRows() = 0;

	// Interprets first `count` of empty rows as data
	virtual void AddRows( int count ) = 0;
	// Frees first `count` of data rows
	virtual void RemoveRows( int count ) = 0;
};

// Rowwise buffer over fully allocated data
// It considers itself empty
// Add rows marks next rows as data
// Free rows moves the pointer of the beginning of the buffer
class CRowwiseWrapper : public IRowwiseBuffer {
public:
	CRowwiseWrapper( float* data, int rowCount, int rowSize );

	// IRowwiseBuffer implementation
	int RowSize() const override { return rowSize; }
	int DataRowIndex() const override { return removedRows; }
	int DataRowCount() const override;
	const float* DataRows() const override;
	int EmptyRowCount() const override { return rowCount - addedRows; }
	float* EmptyRows() override;
	void AddRows( int count ) override;
	void RemoveRows( int count ) override;

private:
	float* firstDataRow;
	const int rowCount;
	const int rowSize;
	int addedRows;
	int removedRows;
};

// Rowwise buffer which allocates data for it
// Always stores data in the beginning of the buffer (moves memory if needed)
class CRowwiseBuffer : public IRowwiseBuffer {
public:
	CRowwiseBuffer( IMathEngine& mathEngine, int rowCount, int rowSize, int fullHeight );

	// IRowwiseBuffer implementation
	int DataRowIndex() const override { return dataRowIndex; }
	int RowSize() const override { return rowSize; }
	int DataRowCount() const override { return dataRowsCount; }
	const float* DataRows() const override;
	int EmptyRowCount() const override;
	float* EmptyRows() override;
	void AddRows( int count ) override;
	void RemoveRows( int count ) override;

private:
	const int rowCount;
	const int rowSize;
	const int fullHeight;
	const int realHeight;
	CFloatHandleVar bufferVar;
	float* bufferPtr;
	float* dataPtr;
	int dataPtrIndex;
	int dataRowsCount;
	int dataRowIndex;
};

//=====================================================================================================================

class CActivationCpuImpl : public IRowwiseCpuImpl, public CRowwiseOperationDesc {
public:
	CActivationCpuImpl( TActivationFunction type, float param0, float param1 );

	int MinInputRowCount() const override { return 1; }

	CBlobDesc Reshape( const CBlobDesc& inputSize ) override;
	int InOperationBufferSize() const override { return 0; }
	int OutputRowCount() const override { return rowCount; }
	int OutputRowSize() const override { return rowSize; }
	bool IsInPlace() const override { return true; }
	CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const override;

private:
	TActivationFunction type;
	float param0;
	float param1;
	int rowCount;
	int rowSize;
};

inline CActivationCpuImpl::CActivationCpuImpl( TActivationFunction type, float param0, float param1 ) :
	type( type ),
	param0( param0 ),
	param1( param1 ),
	rowCount( 0 ),
	rowSize( 0 )
{
}

inline CBlobDesc CActivationCpuImpl::Reshape( const CBlobDesc& inputSize )
{
	rowCount = inputSize.ObjectCount() * inputSize.Height();
	rowSize = inputSize.Width() * inputSize.Channels();
	return inputSize;
}

inline IRowwiseCpuImpl::CProcessingReport CActivationCpuImpl::Process( const float* input, int inputRowIndex,
	int inputRowsAvailable, float* output, int outputRowIndex, int outputRowsAvailable, float* ) const
{
	CProcessingReport result;
	result.OutputRowsCalculated = std::min( outputRowsAvailable, inputRowIndex + inputRowsAvailable - outputRowIndex );
	result.InputRowsMayBeRemoved = outputRowIndex + result.OutputRowsCalculated - inputRowIndex;

	if( inputRowIndex < outputRowIndex ) {
		input += ( outputRowIndex - inputRowIndex ) * rowSize;
	}

	const int dataSize = result.OutputRowsCalculated * rowSize;
	switch( type ) {
		case AF_HSwish:
			vectorHSwish( input, output, dataSize );
			break;
		case AF_ReLU:
			if( param0 <= 0 ) {
				vectorReLU( input, output, dataSize );
			} else {
				vectorReLU( input, output, dataSize, param0 );
			}
			break;
		case AF_Sigmoid:
			vectorSigmoid( input, output, dataSize );
			break;
		case AF_Linear:
			if( input != output ) {
				dataCopy( output, input, dataSize );
			}
			break;
		default:
			ASSERT_EXPR( false );
	}

	return result;
}

} // namespace NeoML
