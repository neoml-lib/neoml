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

#include <CublasDll.h>

namespace NeoML {

// Macros for function loading
#define LOAD_FUNC(Type, Var, NameStr) if((Var = CDll::GetProcAddress<Type>(NameStr)) == 0) return false
#define LOAD_CUBLAS_FUNC(Name) LOAD_FUNC(CCublas::TCublas##Name, functions.Name, "cublas" #Name)
// For the functions with _v2 suffix, define a separate macro
#define LOAD_CUBLAS_FUNCV2(Name) LOAD_FUNC(CCublas::TCublas##Name, functions.Name, "cublas" #Name "_v2")

#if FINE_PLATFORM(FINE_WINDOWS)
static const char* cublasDllName = "cublas64_11.dll";
#elif FINE_PLATFORM(FINE_LINUX)
static const char* cublasDllName = "libcublas.so.11";
#else
#error "Platform is not supported!"
#endif

CCublasDll::CCublasDll()
{
}

CCublasDll::~CCublasDll()
{
	Free();
}

bool CCublasDll::Load()
{
	if( IsLoaded() ) {
		return true;
	}

	if( !CDll::Load( cublasDllName ) ) {
		return false;
	}

	if( !loadFunctions() ) {
		CDll::Free();
		return false;
	}

	return true;
}

void CCublasDll::Free()
{
	if( IsLoaded() ) {
		CDll::Free();
	}
}

// Load all cublas functions used
bool CCublasDll::loadFunctions()
{
	LOAD_CUBLAS_FUNCV2( Create );
	LOAD_CUBLAS_FUNCV2( Destroy );
	LOAD_CUBLAS_FUNC( SetMathMode );
	LOAD_CUBLAS_FUNCV2( SetPointerMode );
	LOAD_CUBLAS_FUNC( SetAtomicsMode );
	LOAD_CUBLAS_FUNCV2( Sdot );
	LOAD_CUBLAS_FUNCV2( Saxpy );
	LOAD_CUBLAS_FUNCV2( Sgemm );
	LOAD_CUBLAS_FUNC( SgemmStridedBatched );
	return true;
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
