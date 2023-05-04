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

} // namespace NeoML
