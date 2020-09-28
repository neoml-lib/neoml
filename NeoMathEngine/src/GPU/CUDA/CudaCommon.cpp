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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <CudaCommon.h>
#include <CudaAssert.h>
#include <MathEngineCommon.h>

#include <mutex>
#include <unordered_map>
#include <thread>

namespace NeoML {

static std::mutex deviceMapMutex;
static std::unordered_map<thread::id, int> deviceMap;

void SetCudaDevice( int deviceNum )
{
	std::thread::id id = std::this_thread::get_id();
	std::lock_guard<std::mutex> lock( deviceMapMutex );
	auto iterator = deviceMap.find( id );
	if( iterator == deviceMap.end() ) {
		deviceMap.insert( std::make_pair( id, deviceNum ) );
		ASSERT_CUDA( cudaSetDevice( deviceNum ) );
	} else if( iterator->second != deviceNum ) {
		iterator->second = deviceNum;
		ASSERT_CUDA( cudaSetDevice( deviceNum ) );
	}
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
