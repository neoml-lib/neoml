/* Copyright © 2023-2024 ABBYY

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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_SSE

namespace NeoML {

namespace Avx512 {

// The minimum vector size recommended for using AVX512 vector functions
static constexpr int VectorMathMinSize = 16;

void dataCopy( float* dst, const float* src, int vectorSize );

void vectorFill( float* result, int vectorSize, float value = 0.f );

void vectorAdd( const float* first, const float* second, float* result, int vectorSize );

void vectorAddValue( const float* first, float* result, int vectorSize, float value );

void vectorMultiply( const float* first, float* result, int vectorSize, float multiplier );

void vectorEltwiseMultiply( const float* first, const float* second, float* result, int vectorSize );

void vectorEltwiseMultiplyAdd( const float* first, const float* second, float* result, int vectorSize );

void vectorReLU( const float* first, float* result, int vectorSize );

void vectorReLU( const float* first, float* result, int vectorSize, float threshold );

void vectorHSwish( const float* first, float* result, int vectorSize );

} // namespace Avx512

} // namespace NeoML

#endif // NEOML_USE_SSE
