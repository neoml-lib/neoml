/* Copyright © 2017-2024 ABBYY

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

#include <MathEngineDnnDropout.h>

namespace NeoML {

CBaseDropoutDesc::CBaseDropoutDesc() :
	Input( nullptr ),
	Output( nullptr ),
	ForwardRate(0.f),
	IsSpatial( false ),
	IsBatchwise( false ),
	Mask( nullptr ),
	IsValid( false ),
	Value( 0.f ),
	Seed( 0 ),
	Threshold( 0 )
{
}

CBaseDropoutDesc::~CBaseDropoutDesc()
{
	if(Mask != nullptr) {
		delete Mask;
	}
	
	if(Input != nullptr) {
		delete Input;
	}

	if(Output != nullptr) {
		delete Output;
	}
}

CSeedDropoutDesc::CSeedDropoutDesc(IMathEngine& mathEngine, bool isMask)
{
	if(isMask) {
		Mask = new CFloatHandleVar(mathEngine, cacheSize);
	}
}

} // namespace NeoML
