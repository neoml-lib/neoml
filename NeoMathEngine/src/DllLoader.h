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

#include <common.h>
#pragma hdrstop

#include <NeoMathEngine/CrtAllocatedObject.h>

#ifdef NEOML_USE_CUDA
#include <cuda_runtime.h>
#include <CudaMathEngine.h>
#include <CublasDll.h>
#include <CusparseDll.h>
#endif

namespace NeoML {

// DLL loading control
class CDllLoader : public CCrtAllocatedObject {
public:
#ifdef NEOML_USE_CUDA
	static CCusparseDll* cusparseDll;
	static CCublasDll* cublasDll;
	static int cudaDllLinkCount;
	static constexpr int CUDA_DLL = 0x1;
#else
	static constexpr int CUDA_DLL = 0x0;
#endif

	static constexpr int ALL_DLL = CUDA_DLL;

	explicit CDllLoader( int dll = ALL_DLL ) : loadedDlls( Load( dll ) ) {}
	~CDllLoader() { Free( loadedDlls ); }

	bool IsLoaded( int dll ) const { return ( loadedDlls & dll ) != 0; }

	static int Load( int dll );
	static void Free( int dll );

private:
	int loadedDlls;
};

} // namespace NeoML
