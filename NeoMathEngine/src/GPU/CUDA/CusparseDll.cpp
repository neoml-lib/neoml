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
#define LOAD_FUNC(Type, Var, NameStr) if((Var = (Type)(void*)CDll::GetProcAddress(NameStr)) == 0) return false
#define LOAD_CUSPARSE_FUNC(Name) LOAD_FUNC(CCusparse::TCusparse##Name, functions.Name, "cusparse" #Name)

// The library name
#if FINE_PLATFORM(FINE_WINDOWS)
static const char* cusparseDllName = "cusparse64_10.dll";
#elif FINE_PLATFORM(FINE_LINUX)
static const char* cusparseDllName = "libcusparse.so.10";
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
	LOAD_CUSPARSE_FUNC( CreateMatDescr );
	LOAD_CUSPARSE_FUNC( DestroyMatDescr );
	LOAD_CUSPARSE_FUNC( SetMatType );
	LOAD_CUSPARSE_FUNC( SetMatIndexBase );
	LOAD_CUSPARSE_FUNC( Scsrmm );
	LOAD_CUSPARSE_FUNC( Scsrmm2 );
	LOAD_CUSPARSE_FUNC( GetErrorString );
	return true;
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
