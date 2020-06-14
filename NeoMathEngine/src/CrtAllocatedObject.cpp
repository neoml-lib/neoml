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

#include <common.h>
#pragma hdrstop

#include <stdlib.h>
#include <NeoMathEngine/CrtAllocatedObject.h>
#include <MathEngineCommon.h>

namespace NeoML {

void* CCrtAllocatedObject::operator new(size_t size)
{
	void* result = malloc(size);
	if( result == 0 ) {
		THROW_MEMORY_EXCEPTION;
	}
	return result;
}

void CCrtAllocatedObject::operator delete(void* ptr)
{
	free(ptr);
}

void* CCrtAllocatedObject::operator new[](size_t size)
{
	void* result = malloc(size);
	if( result == 0 ) {
		THROW_MEMORY_EXCEPTION;
	}
	return result;
}

void CCrtAllocatedObject::operator delete[](void* ptr)
{
	free(ptr);
}

} // namespace NeoML
