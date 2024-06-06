/* Copyright © 2024 ABBYY

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

#ifdef NEOML_USE_CUDA

#include <MemoryHandleInternal.h>

namespace NeoML {

template <typename T>
class CCudaScalarParameter final {
public:
	__host__ CCudaScalarParameter( const CScalarParameter<T>& scalar ) :
		value( scalar.Handle.IsNull() ? scalar.Value : 0 ),
		ptr( scalar.Handle.IsNull() ? nullptr : GetRaw( scalar.Handle ) )
	{}

	__host__ __device__ operator T() const { return ptr == nullptr ? value : *ptr; }

private:
	T value{};
	const T* ptr{};
};

} // namespace NeoML

#endif // NEOML_USE_CUDA
