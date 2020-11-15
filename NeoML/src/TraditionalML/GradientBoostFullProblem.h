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

struct CFloatVectorElement {
	int Index;
	float Value;
};

// The subproblem for building a tree using gradient boosting
class CGradientBoostFullProblem : public IObject {
public:
	CGradientBoostFullProblem( int threadCount, const IMultivariateRegressionProblem* baseProblem,
		const CArray<int>& usedVectors, const CArray<int>& usedFeatures,
		const CArray<int>& featureNumbers );

	// Updates the problem contents
	// Call this method after changing the usedVectors/usedFeatures/featureNumbers arrays
	void Update();

	// Gets the number of vectors in the original problem
	int GetTotalVectorCount() const;

	// Gets the number of features in the original problem
	int GetTotalFeatureCount() const;

	// Gets the number of vectors in the subproblem
	int GetUsedVectorCount() const { return usedVectors.Size(); }

	// Gets the number of features in the subproblem
	int GetUsedFeatureCount() const { return usedFeatures.Size(); }

	// Gets the indices of the features from the original problem that are used in the subproblem
	const CArray<int>& GetUsedFeatureIndexes() const { return usedFeatures; }

	// Checks if the feature is binary
	bool IsUsedFeatureBinary( int feature ) const;

	// Gets the pointer to the start of the given feature values
	const void* GetUsedFeatureDataPtr( int feature ) const;

	// Gets the number of different values of the given feature
	int GetUsedFeatureDataSize( int feature ) const;

protected:
	// delete prohibited
	virtual ~CGradientBoostFullProblem() {}

private:
	const int threadCount; // the number of processing threads
	// The original problem that contains all vectors of the original set
	// The vectors for the subsample will be stored in the usedVectors field (see below)
	const CPtr<const IMultivariateRegressionProblem> baseProblem;
	// The vectors used in the subproblem
	// The array stores the index of each vector in the original set
	// The array length is equal to N * CParams::Subsample, where N is the number of vectors in the original set
	const CArray<int>& usedVectors;
	// The features used in the subproblem
	// The array stores the index of each feature in the original feature set
	// The array length is equal to N * CParams::Subfeature, where N is the total number of features
	const CArray<int>& usedFeatures;
	// Inverse mapping for usedFeatures
	// The array length is the total number of features
	const CArray<int>& featureNumbers;

	CArray<int> featureValueCount; // the number of values for each of the subproblem features
	CArray<bool> isUsedFeatureBinary; // the types of the subproblem features
	CArray<CFloatVectorElement> featureValues; // feature values
	CArray<int> binaryFeatureValues; // binary feature values
	CArray<int> featurePos; // the start of the values of the specific feature in the featureValues array
};

} // namespace NeoML
