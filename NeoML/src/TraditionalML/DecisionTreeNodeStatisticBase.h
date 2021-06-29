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

#include <NeoML/TraditionalML/DecisionTree.h>

namespace NeoML {

// Statistics accumulated in a node
class CDecisionTreeNodeStatisticBase {
public:
	virtual ~CDecisionTreeNodeStatisticBase() {}

	// Adds a vector to the statistics
	virtual void AddVector( int index, const CFloatVectorDesc& vector ) = 0;

	// Finishes accumulating data
	virtual void Finish() = 0;

	// Retrieves the size of accumulated data
	virtual size_t GetSize() const = 0;

	// Gets the optimal split based on the accumulated data
	// Returns false if splitting is not possible
	// featureIndex is the index of the feature by which the node will split
	// values contains the feature values defining the split
	virtual bool GetSplit( CDecisionTree::CParams param,
		bool& isDiscrete, int& featureIndex, CArray<double>& values, double& criterioValue ) const = 0;

	// Gets the predictions according to the accumulated data
	virtual double GetPredictions( CArray<double>& predictions ) const = 0;

	// Returns the number of vectors for which statistics were accumulated
	virtual int GetVectorsCount() const = 0;

	// The node for which statistics are gathered
	virtual CDecisionTreeNodeBase& GetNode() const = 0;
};

} // namespace NeoML
