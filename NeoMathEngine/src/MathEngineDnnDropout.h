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
	virtual void UpdateDesc(const CBlobDesc* input, const CBlobDesc* output, int seed, bool valid) = 0;

	CBlobDesc Input; // input blob descriptor
	CBlobDesc Output; // output blob descriptor
	const float ForwardRate; // the probability that an element is not dropped out
	const bool IsSpatial; // indicates if whole channels are dropped out
	const bool IsBatchwise; // indicates if an element is dropped out of all objects in one batch at the same time
	CFloatHandleVar* Mask; // pointer to mask
	bool IsValid; // is the dropout is valid in current state
	const float Value; // = 1.f / desc.ForwardRate;
	int Seed; // seed for generation mask
	const unsigned Threshold; // = (unsigned int)(desc.ForwardRate * UINT_MAX);

protected:
	CBaseDropoutDesc(float rate, bool isSpatial, bool isBatchwise);
	CBaseDropoutDesc(const CBaseDropoutDesc&) = delete;
	CBaseDropoutDesc& operator=(const CBaseDropoutDesc&) = delete;
};

// Dropout descriptor containing whole mask
struct CMaskDropoutDesc : public CBaseDropoutDesc {
public:
	explicit CMaskDropoutDesc(IMathEngine& mathEngine, float rate, bool isSpatial, bool isBatchwise);
	void UpdateDesc(const CBlobDesc* input, const CBlobDesc* output, int seed, bool valid) override;
	static inline int GetMaskSize(bool isSpatial, bool isBatchwise, const CBlobDesc& input);

private:
	IMathEngine& mathEngine;
};

// Dropout descriptor containing fixed memory for generating mask parts iteratively
struct CSeedDropoutDesc : public CBaseDropoutDesc {
public:
	explicit CSeedDropoutDesc(IMathEngine& mathEngine, float rate, bool isSpatial, bool isBatchwise, bool isMask);
	void UpdateDesc(const CBlobDesc* input, const CBlobDesc* output, int seed, bool valid) override;

	static constexpr int CacheSize = 64;
	static constexpr int MaskAlign = 4;
	static constexpr int NumOfGenerations = (CacheSize + (MaskAlign - 1)) / MaskAlign;
};

inline int CMaskDropoutDesc::GetMaskSize(bool isSpatial, bool isBatchwise, const CBlobDesc& input)
{
	const int objectSize = isSpatial ? input.Channels() : input.ObjectSize();
	const int batchLength = isBatchwise ? input.ObjectCount() : input.BatchLength();
	const int batchWidth = input.ObjectCount() / batchLength;

	return batchWidth * objectSize;
}

} // namespace NeoML
