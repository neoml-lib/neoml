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

#include <CusparseDll.h>

namespace NeoML {

// Macros to load functions
#define LOAD_FUNC(Type, Var, NameStr) if((Var = (Type)CDll::GetProcAddress(NameStr)) == 0) { printf( #NameStr "\n" ); return false; }
#define LOAD_CUSPARSE_FUNC(Name) LOAD_FUNC(CCusparse::TCusparse##Name, functions.Name, "cusparse" #Name)

// The library name
#ifndef NEOML_CUDA_VERSION
#error "NEOML_CUDA_VERSION not set!"
#endif

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

#if FINE_PLATFORM(FINE_WINDOWS)
static const char* cusparseDllName = "cusparse64_" STRINGIFY(NEOML_CUDA_VERSION) ".dll";
#elif FINE_PLATFORM(FINE_LINUX)
static const char* cusparseDllName = "libcusparse.so." STRINGIFY(NEOML_CUDA_VERSION);
#else
#error "Platform is not supported!"
#endif


CCusparseDll::CCusparseDll()
{
}

CCusparseDll::~CCusparseDll()
{
	Free();
}

bool CCusparseDll::Load()
{
	if( IsLoaded() ) {
		return true;
	}

	if( !CDll::Load( cusparseDllName ) ) {
		return false;
	}

	if( !loadFunctions() ) {
		CDll::Free();
		return false;
	}

	return true;
}

void CCusparseDll::Free()
{
	if( IsLoaded() ) {
		CDll::Free();
	}
}

// Load all cusparse library functions used
bool CCusparseDll::loadFunctions()
{
	LOAD_CUSPARSE_FUNC( Create );
	LOAD_CUSPARSE_FUNC( Destroy );
	LOAD_CUSPARSE_FUNC( SetStream );
	LOAD_CUSPARSE_FUNC( CreateCsr );
	LOAD_CUSPARSE_FUNC( DestroySpMat );
	LOAD_CUSPARSE_FUNC( CreateDnMat );
	LOAD_CUSPARSE_FUNC( DestroyDnMat );
	LOAD_CUSPARSE_FUNC( SpMM_bufferSize );
	LOAD_CUSPARSE_FUNC( SpMM );
	return true;
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
