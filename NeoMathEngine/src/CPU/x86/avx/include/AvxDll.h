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
#include <NeoMathEngine/NeoMathEngine.h>
#include <MathEngineCommon.h>
#include <memory>

namespace NeoML {

class IAvxDll : public CCrtAllocatedObject {
public:
	constexpr static char const* GetInstanceFuncName = "GetAvxDllInstance";

	#if FINE_PLATFORM( FINE_WINDOWS )
	constexpr static char const* LibName = "NeoMathEngineAvx.dll";
	#elif FINE_PLATFORM( FINE_LINUX )
	constexpr static char const* LibName = "libNeoMathEngineAvx.so";
	#elif FINE_PLATFORM( FINE_DARWIN )
	constexpr static char const* LibName = "libNeoMathEngineAvx.dylib";
	#else
	#error "Platform is not supported!"
	#endif

	typedef IAvxDll* ( *GetInstanceFunc )( 
		int filterCount, int channelCount, int filterHeight, int filterWidth,
		int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
		int dilationHeight, int dilationWidth, int resultHeight, int resultWidth );
	virtual ~IAvxDll() {}
	virtual bool IsBlobConvolutionAvailable() const = 0;
	virtual void BlobConvolution( int threadCount, const float* sourceData, const float* filterData, const float* freeTermData, float* resultData ) const = 0;

};

}