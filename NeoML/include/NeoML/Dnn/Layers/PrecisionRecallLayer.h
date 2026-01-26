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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Layers/QualityControlLayer.h>

namespace NeoML {

// The quality assessment layer for binary classification
// Used when the network outputs a single number, and the input labels are described by 1 and -1
// The layer has two inputs: the first contains the classification result, the second - the correct class labels
// The layer calculates true positives, true negatives, total positives, and total negatives
class NEOML_API CPrecisionRecallLayer : public CQualityControlLayer {
	NEOML_DNN_LAYER( CPrecisionRecallLayer )
public:
	explicit CPrecisionRecallLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Retrieves the result over the last batch as a 4-number array:
	// true positives, positives total, true negatives, negatives total
	void GetLastResult( CArray<int>& results ) const;

protected:
	void Reshape() override;
	void OnReset() override;
	void RunOnceAfterReset() override;

	virtual int PositivesTotal() const { return accumulated->GetData<int>().GetValueAt( TP_PositivesTotal ); }
	virtual int NegativesTotal() const { return accumulated->GetData<int>().GetValueAt( TP_NegativesTotal ); }
	virtual int PositivesCorrect() const { return accumulated->GetData<int>().GetValueAt( TP_PositivesCorrect ); }
	virtual int NegativesCorrect() const { return accumulated->GetData<int>().GetValueAt( TP_NegativesCorrect ); }

private:
	enum { TP_PositivesCorrect, TP_PositivesTotal, TP_NegativesCorrect, TP_NegativesTotal, TP_Count };
	// Store consts in device memory to avoid excess syncs
	CPtr<CDnnBlob> accumulated;
	CPtr<CDnnBlob> current;
};

NEOML_API CLayerWrapper<CPrecisionRecallLayer> PrecisionRecall();

} // namespace NeoML
