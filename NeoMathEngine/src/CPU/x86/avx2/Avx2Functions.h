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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_SSE

namespace NeoML {

namespace Avx2 {

void dataCopy( float* dst, const float* src, int vectorSize );

void vectorFill( float* result, float value, int vectorSize );

void vectorFill0( float* result, int vectorSize );

void vectorAdd( const float* first, const float* second, float* result, int vectorSize );

void vectorEltwiseMultiply( const float* first, const float* second, float* result, int vectorSize );

void vectorEltwiseMultiplyAdd( const float* first, const float* second, float* result, int vectorSize );

void vectorReLU( const float* first, float* result, int vectorSize );

void vectorReLU( const float* first, float* result, int vectorSize, float threshold );

void vectorHSwish( const float* first, float* result, int vectorSize );

} // namespace Avx2

} // namespace NeoML

#endif // NEOML_USE_SSE
