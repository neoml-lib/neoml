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

#include <NeoML/TraditionalML/GradientBoost.h>

namespace NeoML {

/////////////////////////////////////////////////////////////////////////////////////////
// Base class for different regression tree representations.
// Inherited IRegressionTreeNode interface describes the root node of the tree.
// Child nodes (subtrees) must implement IRegressionTreeNode but may not
// implement CRegressionTree.

class CRegressionTree : public IRegressionTreeNode {
public:
	// Type for prediction result.
	using CPrediction = CFastArray<double, 1>;

	// Calculate prediction.
	virtual void Predict( const CFloatVector& features, CPrediction& result ) const = 0;
	virtual void Predict(
		const CSparseFloatVectorDesc& features, CPrediction& result ) const = 0;

	// Calculates feature usage statistics.
	virtual void CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace NeoML

