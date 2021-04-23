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

#include <NeoML/TraditionalML/Problem.h>

namespace NeoML {

// The subproblem for building a tree with gradient boosting
// The original vectors are transformed into vectors of unique integer identifiers
// The identifier is the number of the histogram bin to which the feature value corresponds
class CGradientBoostFastHistProblem : public IObject {
public:
	// Builds a subproblem from the given data
	CGradientBoostFastHistProblem( int threadCount, int maxBins,
		const IMultivariateRegressionProblem& baseProblem,
		const CArray<int>& usedVectors, const CArray<int>& usedFeatures );

	// Gets the number of vectors used
	int GetUsedVectorCount() const { return usedVectors.Size(); }
	// Gets the pointer to the vector data
	const int* GetUsedVectorDataPtr( int index ) const;
	// Gets the size of the vector data
	int GetUsedVectorDataSize( int index ) const;

	// The length of the function value vector
	int GetValueSize() const { return valueSize; };
	// Gets the number of features
	int GetFeatureCount() const { return nullValueIds.Size(); }
	// Gets the array of features used
	const CArray<int>& GetUsedFeatures() const { return usedFeatures; }
	// Gets the array of identifier beginnings for the given feature
	const CArray<int>& GetFeaturePos() const { return featurePos; }
	// Gets the indices of the features from which the given identifier was obtained
	const CArray<int>& GetFeatureIndexes() const { return featureIndexes; }
	// Gets the cut values for each identifier
	const CArray<float>& GetFeatureCuts() const { return cuts; }
	// Gets the array of identifiers for zero feature values
	const CArray<int>& GetFeatureNullValueId() const { return nullValueIds; }

protected:
	// delete prohibited
	virtual ~CGradientBoostFastHistProblem() {}

private:
	// A feature value
	struct CFeatureValue {
		float Value;
		double Weight;
	};

	// The vectors used
	// For each vector in the subsample, the array contains the index it had in the full sample
	// The array has N * CParams::Subsample elements, where N is the number of vectors in the full sample
	const CArray<int>& usedVectors;
	// The features used
	// For each feature in the subset, the array contains the index it had in the full set of features
	// The array has N * CParams::Subfeature elements, where N is the number of features in the full set
	const CArray<int>& usedFeatures;

	int valueSize; // length of the function value vector
	CArray<int> featurePos; // the identifier positions for this feature
	CArray<int> featureIndexes; // the indices of the feature to which the identifier belongs
	CArray<float> cuts; // the cut values for histograms
	CArray<int> nullValueIds; // the identifiers of the zero feature values
	CArray<int> vectorData; // the vector data
	CArray<int> vectorPtr; // the pointers to the data of the given vector

	void initializeFeatureInfo( int threadCount, int maxBins, const CFloatMatrixDesc& matrix,
		const IMultivariateRegressionProblem& baseProblem );
	void compressFeatureValues( int threadCount, int maxBins, double totalWeight,
		CArray< CArray<CFeatureValue> >& featureValues );
	void buildVectorData( const CFloatMatrixDesc& matrix );
};

} // namespace NeoML
