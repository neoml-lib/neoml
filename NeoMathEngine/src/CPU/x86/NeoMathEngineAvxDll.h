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

#if !FINE_PLATFORM( FINE_IOS )

#include <NeoMathEngine/NeoMathEngine.h>
#include <MathEngineDnnConv.h>
#include <MathEngineDll.h>
#include <avx/include/AvxDll.h>

namespace NeoML {

// Class support dynamic library which is compiled with AVX for performance increasing.
class CNeoMathEngineAvxDll : public CDll {
public:
	CNeoMathEngineAvxDll( const CNeoMathEngineAvxDll& ) = delete;
	CNeoMathEngineAvxDll& operator=( const CNeoMathEngineAvxDll& ) = delete;
	CNeoMathEngineAvxDll( CNeoMathEngineAvxDll&& ) = delete;
	CNeoMathEngineAvxDll& operator=( CNeoMathEngineAvxDll&& ) = delete;

	static CNeoMathEngineAvxDll& GetInstance();
	bool IsAvailable() const;
	std::unique_ptr<IAvxDll> GetAvxDllInst( const CCommonConvolutionDesc& desc );

private:
	bool isLoaded;
	IAvxDll::GetInstanceFunc getAvxDllInstFunc;

	CNeoMathEngineAvxDll();
	~CNeoMathEngineAvxDll() = default;

	static bool isAvxAvailable();
};
}

#endif // !FINE_PLATFORM( FINE_IOS )
