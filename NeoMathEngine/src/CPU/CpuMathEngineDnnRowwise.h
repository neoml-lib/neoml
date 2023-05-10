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

namespace NeoML {

// CPU implementation of rowwise operation
class IRowwiseCpuImpl {
public:
	virtual ~IRowwiseCpuImpl() = default;

	virtual int RequiredRowsCount() const = 0;

	virtual CBlobDesc Reshape( const CBlobDesc& inputSize ) = 0;
	virtual int InOperationBufferSize() const = 0;
	virtual int OutputHeight() const = 0;
	virtual int OutputRowSize() const = 0;

	// The result of single rowwise processing
	struct CProcessingReport {
		int OutputRowsCalculated; // number of output rows calculated during this call
		int InputRowsMayBeRemoved; // number of input rows which are not needed anymore
	};

	virtual CProcessingReport Process( const float* input, int inputRowIndex, int inputRowsAvailable,
		float* output, int outputRowIndex, int outputRowsAvailable, float* buffer ) const = 0;
};

// Interface for buffers for rowwise operations
// First DataRowsCount() rows in buffer are filled with actual data
// Rest of the rows are considered empty
class IRowwiseBuffer {
public:
	virtual ~IRowwiseBuffer() = default;

	// Number of elements in row
	virtual int RowSize() const = 0;

	// Index of first data row in buffer
	virtual int DataRowIndex() const = 0;
	// Number of rows in buffer filled with data
	virtual int DataRowsCount() const = 0;
	// Number of data rows ever appeared in this buffer
	virtual int DataRowsProcessed() const { return DataRowIndex() + DataRowsCount(); }

	// Pointer to the beginning of rows with data
	virtual const float* DataRows() const = 0;

	// Number of empty rows in buffer
	virtual int EmptyRowsCount() const = 0;
	// Pointer after data rows
	virtual float* EmptyRows() = 0;

	// Interprets first `count` of empty rows as data
	virtual void AddRows( int count ) = 0;
	// Frees first `count` of data rows
	virtual void RemoveRows( int count ) = 0;
	// Resets buffer to fully empty and sets data row index to 0
	virtual void Reset() = 0;
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
	int DataRowsCount() const override;
	const float* DataRows() const override;
	int EmptyRowsCount() const override { return rowCount - addedRows; }
	float* EmptyRows() override;
	void AddRows( int count ) override;
	void RemoveRows( int count ) override;
	void Reset() override;

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
	CRowwiseBuffer( IMathEngine& mathEngine, int rowCount, int rowSize );

	// IRowwiseBuffer implementation
	int DataRowIndex() const override { return dataRowIndex; }
	int RowSize() const override { return rowSize; }
	int DataRowsCount() const override { return dataRowsCount; }
	const float* DataRows() const override;
	int EmptyRowsCount() const override { return rowCount - dataRowsCount; }
	float* EmptyRows() override;
	void AddRows( int count ) override;
	void RemoveRows( int count ) override;
	void Reset() override;

private:
	const int rowCount;
	const int rowSize;
	CFloatHandleVar bufferVar;
	float* bufferPtr;
	int dataRowsCount;
	int dataRowIndex;
};

} // namespace NeoML
