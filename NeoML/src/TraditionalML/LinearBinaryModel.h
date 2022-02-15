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

#include <NeoML/TraditionalML/Linear.h>

namespace NeoML {

// The linear binary classifier
class CLinearBinaryModel : public ILinearBinaryModel, public ILinearRegressionModel {
public:
	CLinearBinaryModel() = default;
	CLinearBinaryModel( const CFloatVector& plane, const CSigmoid& sigmoidCoefficients );

	// For serialization
	static CPtr<IModel> Create() { return FINE_DEBUG_NEW CLinearBinaryModel(); }

	// IModel interface methods
	int GetClassCount() const override { return 2; }
	bool Classify( const CFloatVectorDesc& data, CClassificationResult& result ) const override;
	void Serialize( CArchive& archive ) override;

	// ILinearBinaryModel interface methods
	CFloatVector GetPlane() const override { return plane; }
	const CSigmoid& GetSigmoid() const override { return coefficients; }

	// IRegressionModel interface method
	double Predict( const CFloatVectorDesc& data ) const override;

protected:
	~CLinearBinaryModel() override = default; // delete prohibited

private:
	CFloatVector plane; // the base plane
	CSigmoid coefficients; // sigmoid coefficients for estimating probability

	bool classify( double distance, CClassificationResult& result ) const;
};

} // namespace NeoML
