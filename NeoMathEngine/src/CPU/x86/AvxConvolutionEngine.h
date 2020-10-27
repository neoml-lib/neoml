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
#include <MathEngineDll.h>
#include <avx/include/AvxDll.h>

namespace NeoML {

// Class support dynamic library which is compiled with AVX for performance increasing.
class CAvxConvolutionEngine : public CDll {
public:
	CAvxConvolutionEngine( const CAvxConvolutionEngine& ) = delete;
	CAvxConvolutionEngine& operator=( const CAvxConvolutionEngine& ) = delete;
	CAvxConvolutionEngine( CAvxConvolutionEngine&& ) = delete;
	CAvxConvolutionEngine& operator=( CAvxConvolutionEngine&& ) = delete;

	static CAvxConvolutionEngine& GetInstance();
	std::unique_ptr<ISimdConvolutionEngine> InitSimdConvolutionEngine( int filterCount, int channelCount, int filterHeight, int filterWidth );

private:
	bool isLoaded;
	IAvxDllLoader::GetInstanceFunc getAvxDllLoaderFunc;

	CAvxConvolutionEngine();
	~CAvxConvolutionEngine() = default;

	static bool isAvxAvailable();
};
}