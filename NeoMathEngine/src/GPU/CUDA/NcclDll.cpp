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

#ifdef NEOML_USE_NCCL

#include <NcclDll.h>

namespace NeoML {

// Macros for function loading
#define LOAD_FUNC(Type, Var, NameStr) if((Var = CDll::GetProcAddress<Type>(NameStr)) == 0) return false
#define LOAD_NCCL_FUNC(Name) LOAD_FUNC(CNccl::TNccl##Name, functions.Name, "nccl" #Name)

#if FINE_PLATFORM(FINE_LINUX)
static const char* ncclDllName = "libnccl.so";
#else
#error "Platform is not supported!"
#endif

CNcclDll::CNcclDll()
{
}

CNcclDll::~CNcclDll()
{
	Free();
}

bool CNcclDll::Load()
{
	if( IsLoaded() ) {
		return true;
	}

	if( !CDll::Load( ncclDllName ) ) {
		return false;
	}

	if( !loadFunctions() ) {
		CDll::Free();
		return false;
	}

	return true;
}

void CNcclDll::Free()
{
	if( IsLoaded() ) {
		CDll::Free();
	}
}

// Load all cublas functions used
bool CNcclDll::loadFunctions()
{
	LOAD_NCCL_FUNC( CommInitAll );
	LOAD_NCCL_FUNC( CommDestroy );
	LOAD_NCCL_FUNC( AllReduce );
	LOAD_NCCL_FUNC( Broadcast );
	LOAD_NCCL_FUNC( CommInitRank );
	LOAD_NCCL_FUNC( GroupStart );
	LOAD_NCCL_FUNC( GroupEnd );
	LOAD_NCCL_FUNC( GetUniqueId );
	LOAD_NCCL_FUNC( GetErrorString );
	LOAD_NCCL_FUNC( CommGetAsyncError );
	LOAD_NCCL_FUNC( CommAbort );

	return true;
}

} // namespace NeoML

#endif // NEOML_USE_NCCL
