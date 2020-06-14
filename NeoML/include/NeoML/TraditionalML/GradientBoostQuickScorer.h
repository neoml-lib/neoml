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

// The QuickScorer algorithm implementation. See http://ecmlpkdd2017.ijs.si/papers/paperID718.pdf
// A gradient boosting model optimized with the help of this algorithm may perform up to 10 times faster.
// The algorithm is especially efficient if the tree depth is not greater than 6.

#pragma once

#include <NeoML/TraditionalML/GradientBoost.h>

namespace NeoML {

DECLARE_NEOML_MODEL_NAME( GradientBoostQSModelName, "FmlGradientBoostQSModel" )

// Optimized model interface
class NEOML_API IGradientBoostQSModel : public IModel {
public:
	virtual ~IGradientBoostQSModel();

	// Gets the learning rate
	virtual double GetLearningRate() const = 0;

	// Gets the classification results for all tree ensembles [1..k], 
	// with k taking values from 1 to the total number of trees
	virtual bool ClassifyEx( const CSparseFloatVector& data, CArray<CClassificationResult>& results ) const = 0;
	virtual bool ClassifyEx( const CSparseFloatVectorDesc& data, CArray<CClassificationResult>& results ) const = 0;
};

// Optimized regression model interface
class NEOML_API IGradientBoostQSRegressionModel : public IRegressionModel {
public:
	virtual ~IGradientBoostQSRegressionModel();

	// Gets the learning rate
	virtual double GetLearningRate() const = 0;
};

// The QuickScorer algorithm for optimizing a gradient boosting model
class NEOML_API CGradientBoostQuickScorer {
public:
	// Builds a IGradientBoostQSModel based on the given IGradientBoostModel
	CPtr<IGradientBoostQSModel> Build( const IGradientBoostModel& gradientBoostModel );

	// Builds a IGradientBoostQSRegressionModel based on the given IGradientBoostRegressionModel
	CPtr<IGradientBoostQSRegressionModel> BuildRegression( const IGradientBoostRegressionModel& gradientBoostModel );
};

} // namespace NeoML
