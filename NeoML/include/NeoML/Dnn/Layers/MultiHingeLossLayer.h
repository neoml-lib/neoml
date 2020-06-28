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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/LossLayer.h>

namespace NeoML {

// CBaseMultiHingeLossLayer
class NEOML_API CBaseMultiHingeLossLayer : public CLossLayer {
public:
	void Serialize( CArchive& archive ) override;

protected:
	explicit CBaseMultiHingeLossLayer( IMathEngine& mathEngine, const char* name ) : CLossLayer( mathEngine, name ) {}

	virtual void CalculateEltwiseLoss( const CFloatHandle& first, const CFloatHandle& result, int vectorSize) = 0;
	virtual void CalculateEltwiseLossDiff( const CFloatHandle& first, const CFloatHandle& second, const CFloatHandle& result,
		int vectorSize) = 0;

private:
	virtual void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient) override;
	virtual void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstIntHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient) override;
};

///////////////////////////////////////////////////////////////////////////////////

// CMultiHingeLossLayer
class NEOML_API CMultiHingeLossLayer : public CBaseMultiHingeLossLayer {
	NEOML_DNN_LAYER( CMultiHingeLossLayer )
public:
	explicit CMultiHingeLossLayer( IMathEngine& mathEngine ) : CBaseMultiHingeLossLayer( mathEngine, "CCnnMultiHingeLossLayer" ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void CalculateEltwiseLoss( const CFloatHandle& first, const CFloatHandle& result, int vectorSize ) override;
	virtual void CalculateEltwiseLossDiff( const CFloatHandle& first, const CFloatHandle& second, const CFloatHandle& result,
		int vectorSize ) override;
};

///////////////////////////////////////////////////////////////////////////////////

// CMultiSquaredHingeLossLayer
class NEOML_API CMultiSquaredHingeLossLayer : public CBaseMultiHingeLossLayer {
	NEOML_DNN_LAYER( CMultiSquaredHingeLossLayer )
public:
	explicit CMultiSquaredHingeLossLayer( IMathEngine& mathEngine ) : CBaseMultiHingeLossLayer( mathEngine, "CCnnMultiSquaredHingeLossLayer" ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void CalculateEltwiseLoss( const CFloatHandle& first, const CFloatHandle& result, int vectorSize ) override;
	virtual void CalculateEltwiseLossDiff( const CFloatHandle& first, const CFloatHandle& second, const CFloatHandle& result,
		int vectorSize ) override;
};

} // namespace NeoML
