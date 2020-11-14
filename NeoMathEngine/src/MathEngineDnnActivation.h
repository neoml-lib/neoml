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
#include <NeoMathEngine/CrtAllocatedObject.h>

namespace NeoML {

// The general activation function descriptor
struct CCommonActivationDesc : public CActivationDesc {
	TActivationFunction Type;
	float Param1;
	float Param2;
	// These values duplicate floats from above
	// It's for optimization purposes
	CFloatHandleVar Param1Handle;
	CFloatHandleVar Param2Handle;
	// Helper buffer
	CFloatHandleVar Buffer;

	CCommonActivationDesc( IMathEngine& mathEngine, CActivationInfo activation, size_t bufferSize ) :
		Type( activation.Type ),
		Param1( activation.Param1 ),
		Param2( activation.Param2 ),
		Param1Handle( mathEngine ),
		Param2Handle( mathEngine ),
		Buffer( mathEngine, bufferSize )
	{
		Param1Handle.SetValue( activation.Param1 );
		Param2Handle.SetValue( activation.Param2 );
	}
};

} // namespace NeoML
