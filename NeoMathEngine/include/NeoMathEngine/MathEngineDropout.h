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

#pragma once

#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoMathEngine/CrtAllocatedObject.h>

namespace NeoML {

struct CBaseDropoutDesc : public CDropoutDesc {
public:
	virtual ~CBaseDropoutDesc();

	CBlobDesc Input; // input blob descriptor
	CBlobDesc Output; // output blob descriptor
	float ForwardRate; // the probability that an element is not dropped out
	bool IsSpatial; // indicates if whole channels are dropped out
	bool IsBatchwise; // indicates if an element is dropped out of all objects in one batch at the same time
	CFloatHandleVar* Mask; // pointer to mask
	bool IsValid; // is the dropout is valid in current state
	float Value; // = 1.f / desc.ForwardRate;
	int Seed; // seed for generation mask
	unsigned Threshold; // = (unsigned int)(desc.ForwardRate * UINT_MAX);

protected:
	CBaseDropoutDesc();
	CBaseDropoutDesc(const CBaseDropoutDesc&) = delete;
	CBaseDropoutDesc& operator=(const CBaseDropoutDesc&) = delete;
};

} // namespace NeoML
