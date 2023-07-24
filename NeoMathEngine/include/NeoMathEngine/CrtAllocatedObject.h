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

#include <cstddef>
#include <NeoMathEngine/NeoMathEngineDefs.h>

namespace NeoML {

// Base class for all classes in MathEngine
// Uses the standard malloc and free
// All derived classes can be destroyed with the standard delete
// Even global new and delete are redefined
class NEOMATHENGINE_API CCrtAllocatedObject {
public:
	void* operator new(size_t size);
	void operator delete(void* ptr);
	void* operator new[](size_t size);
	void operator delete[](void* ptr);
};

} // namespace NeoML
