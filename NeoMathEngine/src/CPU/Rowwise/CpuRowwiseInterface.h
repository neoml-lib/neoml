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

	// Flag for special operations which support 2 thins:
	//    1. only i'th row of input is needed to calculated i'th row of output
	//    2. calculation may be done in-place
	// E.g. most of the activation functions
	virtual bool IsTrivial() const = 0;

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

} // namespace NeoML
