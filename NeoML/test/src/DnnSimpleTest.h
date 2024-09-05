/* Copyright © 2024 ABBYY

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

namespace NeoMLTest {

class CDnnSimpleTestDummyLearningLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CDnnSimpleTestDummyLearningLayer )
public:
	CPtr<CDnnBlob> ExpectedDiff;
	CPtr<CDnnBlob> ActualDiff;

	explicit CDnnSimpleTestDummyLearningLayer( IMathEngine& mathEngine ) :
		CBaseLayer( mathEngine, "CDnnSimpleTestDummyLearningLayer", true ) {}
	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override { outputBlobs[0]->CopyFrom( inputBlobs[0] ); };
	void BackwardOnce() override;
	void LearnOnce() override;
};

//---------------------------------------------------------------------------------------------------------------------

class CDnnSimpleTestDummyLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CDnnSimpleTestDummyLossLayer )
public:
	CPtr<CDnnBlob> Diff;

	explicit CDnnSimpleTestDummyLossLayer( IMathEngine& mathEngine ) :
		CLossLayer( mathEngine, "CDnnSimpleTestDummyLossLayer" ) {}
	void Serialize( CArchive& archive ) override;

protected:
	void BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle, int vectorSize,
		CConstIntHandle, int, CFloatHandle lossValue, CFloatHandle lossGradient ) override;

	void BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
		CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient ) override
	{ BatchCalculateLossAndGradient( batchSize, data, vectorSize, label, labelSize, lossValue, lossGradient, CFloatHandle{} ); }

	void BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
		CConstFloatHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient,
		CFloatHandle labelLossGradient ) override;
};

} // namespace NeoMLTest
