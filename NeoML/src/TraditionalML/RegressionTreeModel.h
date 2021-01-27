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

#include <NeoML/TraditionalML/Model.h>
#include <NeoML/TraditionalML/GradientBoost.h>

namespace NeoML {

// A regression tree node
class CRegressionTreeModel : public IRegressionTreeModel {
public:
	CRegressionTreeModel();

	// Initializes a leaf node
	void InitLeafNode( const CFloatVector& prediction );
	// Initializes a split node
	void InitSplitNode( CRegressionTreeModel& leftChild, CRegressionTreeModel& rightChild, int feature, double threshold );

	// Gets the node that will be used for prediction
	const CRegressionTreeModel* GetPredictionNode( const CSparseFloatVector& data ) const;
	const CRegressionTreeModel* GetPredictionNode( const CSparseFloatVectorDesc& data ) const;
	const CRegressionTreeModel* GetPredictionNode( const CFloatVector& data ) const;

	// IRegressionTreeModel interface methods
	virtual CPtr<IRegressionTreeModel> GetLeftChild() const { return leftChild.Ptr(); }
	virtual CPtr<IRegressionTreeModel> GetRightChild() const { return rightChild.Ptr(); }
	virtual void GetNodeInfo( CRegressionTreeNodeInfo& result ) const { result = info; }
	virtual void CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const;

	// IMultivariateRegressionModel interface methods
	virtual CFloatVector MultivariatePredict( const CSparseFloatVector& data ) const;
	virtual CFloatVector MultivariatePredict( const CSparseFloatVectorDesc& data ) const;
	virtual CFloatVector MultivariatePredict( const CFloatVector& data ) const;
	virtual void Serialize( CArchive& archive );

protected:
	virtual ~CRegressionTreeModel(); // delete prohibited

private:
	CPtr<CRegressionTreeModel> leftChild; // left child
	CPtr<CRegressionTreeModel> rightChild; // right child
	CRegressionTreeNodeInfo info; // the node information

	void calcFeatureStatistics( int maxFeature, CArray<int>& result ) const;
};

} // namespace NeoML

