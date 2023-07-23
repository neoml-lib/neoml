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

#pragma once

#include <NeoMathEngine/BlobDesc.h>
#include <NeoMathEngine/MemoryHandle.h>

namespace NeoML {

// CUDA implementation of rowwise operation
class ICudaRowwiseImpl {
public:
	virtual ~ICudaRowwiseImpl() = default;

	// Must be called before inference
	// Returns the size of output of this operation
	virtual CBlobDesc Reshape( const CBlobDesc& inputSize ) = 0;

	// Number of elements in operation output
	virtual int OutputSize() const = 0;

	// Flag for special operations which can be calculated in-place
	// E.g. most of the activation functions
	virtual bool IsInPlace() const = 0;

	// It isn't effective to compute anything rowwise on GPU
	// That's why it's just a stub which accepts whole input and output
	virtual void Process( const CFloatHandle& input, const CFloatHandle& output ) const = 0;
};

} // namespace NeoML
