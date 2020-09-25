/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <CpuMathEngine.h>
#include <MemoryHandleInternal.h>

// Wraps the stack allocator in a template
// The template does not use NeoML structures, 
// so it can be reused in other projects (for example, you may want to compare performance)
struct CTmpMemoryHandler : public NeoML::CFloatHandleStackVar {
	CTmpMemoryHandler( NeoML::CCpuMathEngine* engine, size_t size ) :
		NeoML::CFloatHandleStackVar( *engine, size )
	{
	}
	float* get()
	{
		return GetRaw( GetHandle() );
	}
};

// Sets the result to 0; the micro-kernel product works as +=
// so the result should be set to 0 in some of the functions
static void nullify( float* data, size_t height, size_t width, size_t rowSize )
{
	float* last = data + rowSize * height;
	for( float* c = data; c < last; c += rowSize ) {
		memset( c, 0, width * sizeof( float ) );
	}
}

// Fills the matrix with zeros (for when the matrix is stored in a single memory block)
static void nullify( float* data, size_t height, size_t width )
{
	memset( data, 0, height * width * sizeof( float ) );
}

