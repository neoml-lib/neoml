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

#include <NeoMathEngine/NeoMathEngineException.h>
#include <NeoMathEngine/MemoryHandle.h>

namespace NeoML {

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

//---------------------------------------------------------------------------------------------------------------------

inline CRowwiseWrapper::CRowwiseWrapper( float* data, int rowCount, int rowSize ) :
	firstDataRow( data ),
	rowCount( rowCount ),
	rowSize( rowSize ),
	addedRows( 0 ),
	removedRows( 0 )
{}

inline int CRowwiseWrapper::DataRowCount() const
{
	PRESUME_EXPR( addedRows >= removedRows );
	return addedRows - removedRows;
}

inline const float* CRowwiseWrapper::DataRows() const
{
	PRESUME_EXPR( DataRowCount() > 0 );
	return firstDataRow;
}

inline float* CRowwiseWrapper::EmptyRows()
{
	PRESUME_EXPR( addedRows < rowCount );
	return firstDataRow + DataRowCount() * rowSize;
}

inline void CRowwiseWrapper::AddRows( int count )
{
	PRESUME_EXPR( count > 0 );
	addedRows += count;
	PRESUME_EXPR( addedRows <= rowCount );
}

inline void CRowwiseWrapper::RemoveRows( int count )
{
	PRESUME_EXPR( count > 0 );
	PRESUME_EXPR( count <= DataRowCount() );
	removedRows += count;
	firstDataRow += count * rowSize;
	PRESUME_EXPR( removedRows <= addedRows );
}

//---------------------------------------------------------------------------------------------------------------------

CRowwiseBuffer::CRowwiseBuffer( IMathEngine& mathEngine, int rowCount, int rowSize, int fullHeight ) :
	rowCount( rowCount ),
	rowSize( rowSize ),
	fullHeight( fullHeight ),
	realHeight( std::min( fullHeight, 2 * rowCount ) ),
	bufferVar( mathEngine, realHeight * rowSize ),
	bufferPtr( GetRaw( bufferVar.GetHandle() ) ),
	dataPtr( bufferPtr ),
	dataPtrIndex( 0 ),
	dataRowsCount( 0 ),
	dataRowIndex( 0 )
{
}

const float* CRowwiseBuffer::DataRows() const
{
	PRESUME_EXPR( dataRowsCount > 0 );
	return dataPtr;
}

int CRowwiseBuffer::EmptyRowCount() const
{
	return std::min( rowCount - dataRowsCount, fullHeight - ( dataRowIndex + dataRowsCount ) );
}

float* CRowwiseBuffer::EmptyRows()
{
	PRESUME_EXPR( EmptyRowCount() > 0 );
	return dataPtr + dataRowsCount * rowSize;
}

void CRowwiseBuffer::AddRows( int count )
{
	PRESUME_EXPR( count > 0 );
	PRESUME_EXPR( count <= EmptyRowCount() );
	dataRowsCount += count;
	PRESUME_EXPR( dataRowsCount <= rowCount );
	PRESUME_EXPR( dataRowsCount + dataPtrIndex <= realHeight );
}

void CRowwiseBuffer::RemoveRows( int count )
{
	PRESUME_EXPR( count > 0 );
	PRESUME_EXPR( count <= dataRowsCount );
	dataRowsCount -= count;
	dataRowIndex += count;
	dataPtr += count * rowSize;
	dataPtrIndex += count;
	if( dataPtrIndex + rowCount > realHeight && dataRowIndex + ( rowCount - dataPtrIndex ) < fullHeight ) {
		if( dataRowsCount > 0 ) {
			// Move remaining data rows to the beginning of buffer
			// HACK: we know that data copying work sequentially that's why we don't need to check for the overlap
			dataCopy( bufferPtr, dataPtr, dataRowsCount * rowSize );
		}
		dataPtr = bufferPtr;
		dataPtrIndex = 0;
	}
}

} // namespace NeoML
