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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#include <MathEngineDll.h>

namespace NeoML {

class ISimdMathEngine;
class IMathEngine;

// The dynamic link simd library
class CAvxDll : public CDll {
public:
	CAvxDll();
	~CAvxDll();

	// Loads the library
	bool Load();

	// Unloads the library
	void Free();

	ISimdMathEngine* CreateSimdMathEngine( IMathEngine* mathEngine, int threadCount );

private:
	constexpr static char const* CreateSimdMathEngineFuncName = "CreateSimdMathEngine";
	typedef ISimdMathEngine* ( *GetSimdMathEngineFunc )( IMathEngine* mathEngine, int threadCount );

	GetSimdMathEngineFunc createSimdMathEngineFunc;

	bool loadFunctions();
	static bool isAvxAvailable();
};

} // namespace NeoML