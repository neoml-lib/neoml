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

	// The minimum number of input rows required for correct work of this operation
	// Usually means the number of input rows required to calculate 1 row of output
	virtual int MinInputRowCount() const = 0;

	// Must be called before inference
	// Returns the size of output of this operation
	virtual CBlobDesc Reshape( const CBlobDesc& inputSize ) = 0;

	// The size of buffer needed during calculation
	// Buffer of equal or greater size must be provided as `buffer` parameter in IRowwiseCpuImpl::Process
	// The data won't be saved between different IRowwiseCpuImpl::Process calls
	// If operation needs dedicated buffer which saves data between different calls
	// it should allocate and manage this buffer by itself (e.g. allocate during Reshape)
	virtual int InOperationBufferSize() const = 0;

	// The number of rows in output
	// Usually equal to outputDesc.ObjectCount() * outputDesc.Height()
	// where outputDesc is the result of last Reshape call
	virtual int OutputRowCount() const = 0;

	// The size of single output row
	// Usually equal to outputDesc.Width() * outputDesc.Depth() * outputDesc.Channels()
	// where outputDesc is the result of last Reshape call
	virtual int OutputRowSize() const = 0;

	// Flag for special operations which can be calculated in-place
	// It means that it always able to process all the input rows provided
	// and may overwrite its input
	// E.g. most of the activation functions
	virtual bool IsInPlace() const = 0;

	// The result of single rowwise processing
	struct CProcessingReport {
		int OutputRowsCalculated = 0; // number of output rows calculated during this call
		int InputRowsMayBeRemoved = 0; // number of input rows which are not needed anymore by this operation
	};

	// Processes [inputRowIndex; inputRowIndex + inputRowsAvailable) rows of the input
	// and calculate up to [outputRowIndex; outputRowIndex + outputRowsAvailable) rows of the input
	// Return the report with information about how many rows of output were calculated
	// and how many rows of input won't be needed by this operation anymore
	virtual CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const = 0;
};

//=====================================================================================================================

// Interface for buffers for rowwise operations
//
// It contains DataRowCount() rows of the full blob starting from DataRowIndex()'th row
// (Full blob contains ObjectCount() * Height() rows)
//
// In addition it has space for the next EmptyRowCount() rows of full blob
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
	// Which means it increases the number of DataRowCount() and reduces the EmptyRowCount()
	virtual void AddRows( int count ) = 0;
	// Frees first `count` of data rows
	// which means it increase DataRowIndex(), reduces DataRowCount() and increases EmptyRowCount()
	virtual void RemoveRows( int count ) = 0;
};

// Rowwise buffer over previously allocated data
class CRowwiseWrapper : public IRowwiseBuffer {
public:
	// data - pointer to the beginning of the buffer
	// rowCount - number of rows in the buffer
	// rowSize - size of one row
	// After the construction it considers itself as a buffer without data
	// which means DataRowCount() is 0 and EmptyRowCount() is rowCount
	// It may allocate more than rowCount in order to reduce number of ::memmove calls
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
	// Pointer to the beginning of data rows
	// RemoveRows moves this pointer
	float* firstDataRow;
	// The initial number of empty rows in buffer
	const int rowCount;
	// The size of a signle row
	const int rowSize;
	// Number of data rows added to buffer during whole lifetime (never decreases)
	int addedRows;
	// Number of data rows removed from buffer during whole lifetime
	// (never decreases, always less or equal to the addedRows)
	int removedRows;
};

// Rowwise buffer which contains only slice of full blob
// Allocates and manages memory by itself
class CRowwiseBuffer : public IRowwiseBuffer {
public:
	// It guarantees to allocate at least rowCount rows and at most fullHeight rows
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
	// Number of rows this buffer is expected to have
	const int rowCount;
	// Size of a single row
	const int rowSize;
	// Number of rows in full blob (ObjectCount() * Height())
	const int fullHeight;
	// Number of rows actually allocated for this buffer, somewhere in [rowCount; fullHeight]
	const int realHeight;
	// MathEngine variable which contains the allocated memory
	CFloatHandleVar bufferVar;
	// Pointer to the beginning of the allocated memory
	float* const bufferPtr;
	// Pointer to the memory where data rows are contained
	float* dataPtr;
	// Number of rows between bufferPtr and dataPtr
	// (distance between the beginning of the buffer and the memory where data currently starts)
	int dataPtrIndex;
	// Number of data rows in buffer
	int dataRowsCount;
	// Index of first data row relative to the full blob [0; fullHeight)
	int dataRowIndex;
};

//=====================================================================================================================

class CRowwiseActivation : public IRowwiseCpuImpl, public CRowwiseOperationDesc {
public:
	CRowwiseActivation( TActivationFunction type, float param0, float param1 );

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

inline CRowwiseActivation::CRowwiseActivation( TActivationFunction type, float param0, float param1 ) :
	type( type ),
	param0( param0 ),
	param1( param1 ),
	rowCount( 0 ),
	rowSize( 0 )
{
}

inline CBlobDesc CRowwiseActivation::Reshape( const CBlobDesc& inputSize )
{
	rowCount = inputSize.ObjectCount() * inputSize.Height();
	rowSize = inputSize.Width() * inputSize.Channels();
	return inputSize;
}

inline IRowwiseCpuImpl::CProcessingReport CRowwiseActivation::Process( const float* input, int inputRowIndex,
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
		case AF_HardSigmoid:
			vectorHardSigmoid( input, output, param0, param1, dataSize );
			break;
		case AF_HSwish:
			vectorHSwish( input, output, dataSize );
			break;
		case AF_Linear:
			if( param0 != 1.f ) {
				vectorMultiply( input, output, param0, dataSize );
				input = output;
			}
			if( param1 != 0.f ) {
				vectorAddValue( input, output, dataSize, param1 );
				input = output;
			}
			if( input != output ) {
				// Corner case: Linear( 1, 0 ), not in-place
				dataCopy( output, input, dataSize );
			}
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
		default:
			ASSERT_EXPR( false );
	}

	return result;
}

//=====================================================================================================================

// Index of first input row needed to calculate outputRowIndex'th row of output
int RowwiseConvFirstInputRow( int outputRowIndex, int inputImageHeight, int outputImageHeight,
	int strideHeight, int paddingHeight );

// Calculates how many output rows can be calculated with the given data
// and how many input rows can be released after that
IRowwiseCpuImpl::CProcessingReport RowwiseConvProcessingReport( int inputRowIndex, int inputRowsAvailable,
	int outputRowIndex, int outputRowsAvailable, int inputImageHeight, int outputImageHeight,
	int filterHeight, int paddingHeight, int strideHeight, int dilationHeight );

} // namespace NeoML
