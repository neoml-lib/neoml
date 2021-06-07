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

#include <RegressionTree.h>

namespace NeoML {

// A regression tree implementation using nodes objects linked with pointers.
class CLinkedRegressionTree : public CRegressionTree {
public:
	CLinkedRegressionTree();

	// Initializes a leaf node
	void InitLeafNode( double prediction );
	void InitLeafNode( const CArray<double>& prediction );
	// Initializes a split node
	void InitSplitNode(
		CLinkedRegressionTree& leftChild, CLinkedRegressionTree& rightChild,
		int feature, double threshold );

	// Gets the node that will be used for prediction
	template<typename TVector>
	const CLinkedRegressionTree* GetPredictionNode( const TVector& data ) const;

	// CRegressionTree methods implementation.
	virtual void Predict(
		const CFloatVector& features, CPrediction& result ) const override;
	virtual void Predict(
		const CFloatVectorDesc& features, CPrediction& result ) const override;
	virtual double Predict( const CFloatVector& features ) const override;
	virtual double Predict( const CFloatVectorDesc& features ) const override;
	virtual void CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const override;

	// IRegressionTreeNode interface methods
	virtual CPtr<const IRegressionTreeNode> GetLeftChild() const override
		{ return leftChild.Ptr(); }
	virtual CPtr<const IRegressionTreeNode> GetRightChild() const override
		{ return rightChild.Ptr(); }
	virtual void GetNodeInfo( CRegressionTreeNodeInfo& result ) const override { result = info; }

	virtual void Serialize( CArchive& archive ) override;

protected:
	virtual ~CLinkedRegressionTree(); // delete prohibited

private:
	CPtr<CLinkedRegressionTree> leftChild; // left child
	CPtr<CLinkedRegressionTree> rightChild; // right child
	CRegressionTreeNodeInfo info; // the node information

	template<typename TVector>
	void predict( const TVector& features, CPrediction& result ) const;
	template<typename TVector>
	double predict( const TVector& features ) const;

	void calcFeatureStatistics( int maxFeature, CArray<int>& result ) const;
};

} // namespace NeoML

