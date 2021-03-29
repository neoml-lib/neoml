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

#include <limits>
#include <RegressionTree.h>

namespace NeoML {

/////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class CCompactRegressionTree : public CRegressionTree {
public:
	CCompactRegressionTree() {}
	explicit CCompactRegressionTree( const IRegressionTreeNode* source );

	// Actual implementation of IRegressionTreeNode for this class and CNodeWrapper,
	CPtr<const IRegressionTreeNode> GetLeftChild( int nodeIndex ) const;
	CPtr<const IRegressionTreeNode> GetRightChild( int nodeIndex ) const;
	void GetNodeInfo( int nodeIndex , CRegressionTreeNodeInfo& info ) const;

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
		{ return GetLeftChild(0); }
	virtual CPtr<const IRegressionTreeNode> GetRightChild() const override
		{ return GetRightChild(0); }
	virtual void GetNodeInfo( CRegressionTreeNodeInfo& info ) const override
		{ GetNodeInfo( 0, info ); }

	virtual void Serialize( CArchive& archive ) override;

	// Feature number cannot exceeed this.
	static const T MaxFeature = (std::numeric_limits<T>::max()) - 2;
	// Node index within the `nodes` array cannot exceed this.
	static const uint16_t MaxNodeIndex = UINT16_MAX - 1;

private:

	// Describes tree node.
	struct CNode {
		// For non-leaf node the index of the feature incremented by one.
		// For leaf node is zero.
		T FeaturePlusOne = 0;
		// For non-leaf node the index of the right child within the `nodes` array.
		// Left child is always stored immediatelly after its parent.
		uint16_t RightChildIndex = 0;

		// For non-leaf node the threshold feature value (scalar).
		// For leaf node the value of regression function (scalar or vector).
		union {
			// Single value resides here.
			float Resident;
			// Multiple values are stored in `nonresidentValues` array
			// starting from `NonresidentIndex` element.
			int32_t NonresidentIndex = 0;
		} Value;
	};

	// Node wrapper implementing IRegressionTreeNode.
	class CNodeWrapper;

	// All tree nodes. Zero element is root node.
	CArray<CNode> nodes;
	// Storage for multivariate regression function values.
	CArray<float> nonresidentValues;
	// Storage for on-demand created wrappers.
	mutable CObjectArray<const IRegressionTreeNode> wrappers;
	// The size of value stored in leaf nodes.
	int predictionSize = NotFound;

	virtual ~CCompactRegressionTree() override = default;

	void importNodes( const IRegressionTreeNode* source );

	CPtr<const IRegressionTreeNode> getWrapper( int nodeIndex ) const;

	template<typename TVector>
	void predict( const TVector& features, CPrediction& result ) const;
	template<typename TVector>
	const float* predict( const TVector& features ) const;
};

typedef CCompactRegressionTree<uint16_t> CCompact16RegressionTree;
typedef CCompactRegressionTree<uint32_t> CCompact32RegressionTree;

/////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class CCompactRegressionTree<T>::CNodeWrapper : public IRegressionTreeNode {
public:
	CNodeWrapper( const CCompactRegressionTree& _tree, int _index ) :
		tree( _tree ),
		index( _index )
	{}

	virtual CPtr<const IRegressionTreeNode> GetLeftChild() const override
		{ return tree.GetLeftChild( index ); }
	virtual CPtr<const IRegressionTreeNode> GetRightChild() const override
		{ return tree.GetRightChild( index ); }
	virtual void GetNodeInfo( CRegressionTreeNodeInfo& info ) const override
		{ tree.GetNodeInfo( index, info ); }

private:
	// Parent tree object.
	const CCompactRegressionTree& tree;
	// Index within `tree.nodes`.
	const int index;
};

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace NeoML

