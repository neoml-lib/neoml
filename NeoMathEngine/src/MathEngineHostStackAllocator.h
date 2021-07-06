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

#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoMathEngine/CrtAllocatedObject.h>
#include <MathEngineAllocator.h>
#include <mutex>
#include <unordered_map>
#include <thread>

using namespace std;

namespace NeoML {

class CHostStackMemoryManager;

// Host memory stack implementation for MathEngine
class CHostStackAllocator : public CCrtAllocatedObject {
public:
	explicit CHostStackAllocator( int memoryAlignment );
	~CHostStackAllocator();

	void CleanUp();

	void* Alloc( size_t size );
	void Free( void* ptr );

private:
	const int memoryAlignment;
	std::mutex mutex;
	std::unordered_map< thread::id, CHostStackMemoryManager*,
		hash<thread::id>, equal_to<thread::id>, CrtAllocator< pair<const thread::id, CHostStackMemoryManager*> > > stackManagers;
};

} // namespace NeoML
