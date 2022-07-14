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

const int MaxQSLeavesCount = 64; // maximum number of leaves in a subtree that can be optimized
const int MaxTreesCount = 32767; // maximum supported number of trees in an ensemble

const unsigned char PM_Inverted = 1; // the node is inverted
const unsigned char PM_LeftLeaf = 2; // the left child is a leaf in the optimized subtree
const unsigned char PM_RightLeaf = 4; // the right child is a leaf in the optimized subtree

// The descriptor of an non-leaf node of the subtree to be optimized
struct CQSNode {
	unsigned __int64 Mask; // the false nodes mask: has zeros in the positions where the vector cannot possibly belong if the node criterion is false
	float Threshold; // split threshold
	short Tree; // the index of the tree in the ensemble
	char Order; // the order of depth first traversal
	unsigned char PropertiesMask; // the node properties (inverted or not, is either of the children a leaf in the optimized subtree)

	CQSNode( unsigned __int64 mask, float threshold, int tree, int order, unsigned char propertiesMask );
};

inline CQSNode::CQSNode( unsigned __int64 mask, float threshold, int tree, int order, unsigned char propertiesMask ) :
	Mask( mask ),
	Threshold( threshold ),
	Tree( static_cast<short>( tree ) ),
	Order( static_cast<char>( order ) ),
	PropertiesMask( propertiesMask )
{
	NeoAssert( tree <= MaxTreesCount );
	NeoAssert( order < MaxQSLeavesCount ); // always fewer nodes than leaves
}

//------------------------------------------------------------------------------------------------------------

// The description of a leaf in the optimized subtree
// Stores either a subtree or a value of a leaf in the original tree
struct CQSLeaf {
	float Value; // the value of a leaf
	int SimpleNodeIndex; // the index of the non-optimized subtree root in the simpleNodes array

	explicit CQSLeaf( float value ) : Value( value ), SimpleNodeIndex( NotFound ) {};
	explicit CQSLeaf( int rootIndex ) : Value( 0 ), SimpleNodeIndex( rootIndex ) {};
};

//------------------------------------------------------------------------------------------------------------

// The description of a node not in the optimized subtree
struct CSimpleNode {
	int Feature; // the index of split feature; if the node is a leaf - NotFound (-1)
	float Value; // the value in the node (splitting threshold or the prediction value)
	int RightChild; // the index of the right child; its left child always has the index of the right + 1

	CSimpleNode( int feature, float value, int rightChild ) : Feature( feature ), Value( value ), RightChild( rightChild ) {}
	explicit CSimpleNode( float value ) : Feature( NotFound ), Value( value ), RightChild( NotFound ) {}
};

// The offset to the optimized nodes that use this feature for splitting
struct CQSNodeOffset {
	CInterval Less;
	CInterval More;

	CQSNodeOffset() : Less( NotFound, NotFound ), More( NotFound, NotFound ) {}
};

//------------------------------------------------------------------------------------------------------------

class IQsSerializer;

// Optimized trees ensemble
class CGradientBoostQSEnsemble {
public:
	// Builds the ensemble from an unoptimized trees ensemble
	void Build( const CGradientBoostEnsemble &treeModel );

	// Prediction methods
	double Predict( const CFloatVectorDesc& data ) const;

	// The prediction method that uses only the trees in the 0 to lastTreeIndex range
	double Predict( const CFloatVectorDesc& data, int lastTreeIndex ) const;

	// Gets the number of trees in the ensemble
	int GetTreesCount() const { return treeQsLeavesOffsets.Size(); };

	// Serialization
	friend CArchive& operator<<( CArchive& archive, const CGradientBoostQSEnsemble& block );
	friend CArchive& operator>>( CArchive& archive, CGradientBoostQSEnsemble& block );

private:
	// Optimized nodes descriptions, sorted by splitting feature
	// The nodes that use ith feature are in the ranges featureQsNodesOffsets[i].Less and featureQsNodesOffsets[i].More
	CArray<CQSNode> qsNodes;
	CMap<int, CQSNodeOffset> featureQsNodesOffsets; // offsets to optimized nodes that use the same feature
	 // Optimized subtree leaves descriptions, sorted by subtree
	CArray<CQSLeaf> qsLeaves; // the leaves of the i subtree start from treeQsLeavesOffsets[i] index
	CArray<int> treeQsLeavesOffsets; // offsets to optimized subtree leaves for the specified tree in the ensemble
	CArray<CSimpleNode> simpleNodes; // the descriptions of the nodes that are not in optimized subtrees

	void store( CArchive& archive ) const;
	void storeQSNode( IQsSerializer& serializer, const CArray<int>& links, const CArray<int>& features,
		int& leafIndex, int& nodeIndex ) const;
	void storeQSLeaf( IQsSerializer& serializer, int& leafIndex ) const;
	void storeSimpleNode( IQsSerializer& serializer, int index ) const;
	void buildNodesFeatures( CArray<int>& features ) const;

	void load( CArchive& archive );
	void loadQSNode( IQsSerializer& serializer, int treeId,
		int& orderId, bool& isQsLeaf, unsigned __int64& mask, CArray<int>& features );
	void loadQSLeaf( IQsSerializer& serializer, int featureIndex, float threshold );
	void loadSimpleSubtree( IQsSerializer& serializer, int featureIndex, float threshold );
	void buildFeatureNodesOffsets( const CArray<int>& features );

	void processFeature( int feature, float value, CFastArray<unsigned __int64, 512>& bitvectors ) const;
	double calculateScore( const CFloatVectorDesc& data, const CFastArray<unsigned __int64, 512>& bitvectors, int lastTreeIndex ) const;
};

} // namespace NeoML
