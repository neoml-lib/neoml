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

#ifdef NEOML_USE_CUDA

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <MathEngineDll.h>
#include <CublasFunctions.h>

namespace NeoML {

// A dynamic link library called cublas
// Used for linear algebra operations
class CCublasDll : public CDll {
public:
	CCublasDll();
	virtual ~CCublasDll();

	// Loads the library
	bool Load();

	// Checks if the library has been loaded
	bool IsLoaded() const { return CDll::IsLoaded(); }

	// Gets the structure that exposes the cublas functions used in MathEngine
	const CCublas* GetFunctions() const { return IsLoaded() ? &functions : 0; }

	// Unloads the library
	void Free();

private:
	CCublas functions; // the structure that exposes the cublas functions used in MathEngine

	bool loadFunctions();
};

} // namespace NeoML

#endif // NEOML_USE_CUDA
