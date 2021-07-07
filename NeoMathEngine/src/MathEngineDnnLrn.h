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

struct CMathEngineLrnDesc : public CLrnDesc {
	CMathEngineLrnDesc( const CBlobDesc& source, int windowSize, float bias, float alpha, float beta ) :
		Source( source ), WindowSize( windowSize ), Bias( bias ), Alpha( alpha ), Beta( beta ) {}

	CBlobDesc Source;
	int WindowSize;
	float Bias;
	float Alpha;
	float Beta;
};

} // namespace NeoML
