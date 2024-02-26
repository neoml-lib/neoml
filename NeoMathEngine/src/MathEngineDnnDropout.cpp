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

CBaseDropoutDesc::CBaseDropoutDesc(float rate, bool isSpatial, bool isBatchwise) :
	Input( CT_Float ),
	Output( CT_Float ),
	ForwardRate( 1.f - rate ),
	IsSpatial(isSpatial),
	IsBatchwise(isBatchwise),
	Mask( nullptr ),
	IsValid( false ),
	Value(1.f / (1.f - rate)),
	Seed( 0 ),
	Threshold((unsigned int)(ForwardRate* UINT_MAX))
{
	ASSERT_EXPR(rate >= 0.f && rate < 1.f);
}

CBaseDropoutDesc::~CBaseDropoutDesc()
{
	if(Mask != nullptr) {
		delete Mask;
	}
}

CMaskDropoutDesc::CMaskDropoutDesc(IMathEngine& mathEngine, float rate, bool isSpatial, bool isBatchwise) :
	CBaseDropoutDesc(rate, isSpatial, isBatchwise),
	mathEngine( mathEngine )
{
}

void CMaskDropoutDesc::UpdateDesc(const CBlobDesc* input, const CBlobDesc* output, int seed, bool valid)
{
	if(IsValid == valid) {
		return;
	}

	IsValid = valid;
	if(Mask != nullptr) {
		delete Mask;
		Mask = nullptr;
	}

	if (valid) {
		ASSERT_EXPR(input != nullptr);
		ASSERT_EXPR(output != nullptr);

		Input = *input;
		Output = *output;

		Seed = seed;

		Mask = new CFloatHandleVar(mathEngine, CMaskDropoutDesc::GetMaskSize(IsSpatial, IsBatchwise, *input));
		mathEngine.VectorFillBernoulli(Mask->GetHandle(), ForwardRate, Mask->Size(),
			1.f / ForwardRate, Seed);
	}
}

CSeedDropoutDesc::CSeedDropoutDesc(IMathEngine& mathEngine, float rate, bool isSpatial, bool isBatchwise, bool isMask) :
	CBaseDropoutDesc(rate, isSpatial, isBatchwise)
{
	if (isMask) {
		Mask = new CFloatHandleVar(mathEngine, CacheSize);
	}
}

void CSeedDropoutDesc::UpdateDesc(const CBlobDesc* input, const CBlobDesc* output, int seed, bool valid)
{
	if (IsValid == valid) {
		return;
	}

	IsValid = valid;;
	if (valid) {
		Seed = seed;

		ASSERT_EXPR(input != nullptr);
		ASSERT_EXPR(output != nullptr);

		Input = *input;
		Output = *output;
	}
}

} // namespace NeoML
