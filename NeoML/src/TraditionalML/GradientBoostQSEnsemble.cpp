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

// QuickScorer implementation. Reference: http://ecmlpkdd2017.ijs.si/papers/paperID718.pdf
//		  This algorithm speeds up the leaf search in trees trained by gradient boosting.
//		Speeding up is possible for up to 64 leaves per tree. If a tree has more leaves, 
//		a subtree with 64 leaves is chosen to be optimized. If a leaf in the optimized subtree is not a leaf in the whole tree 
//		the search will continue from there using the standard algorithm.
//		  For each non-leaf node of the subtree the algorithm calculates a 64-bit bit mask, with each bit encoding one of the leaves.
//		It puts zeros in the positions where we definitely cannot arrive if this node criterion is not fulfilled and ones in other positions.
// 		To get the leaf for the vector take all nodes with unfulfilled criteria and calculate bitwise AND for their bit masks. 
//		The index of the required leaf is the index of the highest nonzero bit in the result.
//		  In this implementation, the leaves are numbered left to right (the masks are inverted compared with the reference paper), 
//		so we are looking for the lowest nonzero bit. 
//		  The subtrees are also turned around, so that for zero value the left subtree is taken. 
//		So we don't need to pass zero feature values to the algorithm because the bit mask does not change anyway. 
//		For inverted nodes the < operator is changed to >=, so we use two mask sets: for inverted and non-inverted nodes, 
//		search in both sets and merge the results.

#include <common.h>
#pragma hdrstop

#include <GradientBoostQSEnsemble.h>

namespace NeoML {

// Returns the lowest nonzero bit index in the binary representation of a number
inline static int findLowestBitIndex( unsigned __int64 value )
{
#if FINE_64_BIT
	unsigned long result = 0;
	_BitScanForward64( &result, value );
	return static_cast<int>( result );
#else
	const DWORD low = value & 0xFFFFFFFF;
	unsigned long result = 0;
	if( _BitScanForward( &result, low ) == 0 ) {
		const DWORD hi = value >> 32;
		if( _BitScanForward( &result, hi ) ) {
			return static_cast<int>(result) + 32;
		}
		return 0;
	}
	return static_cast<int>(result);
#endif
}

// Finds the nodes that will be used for leaves of the optimized subtree
inline static void findQsLeaves( const IRegressionTreeModel* root, CHashTable<const IRegressionTreeModel*>& qsLeaves )
{
	CFastArray<const IRegressionTreeModel*, MaxQSLeavesCount * 2> nextLevel; // next level nodes
	CFastArray<const IRegressionTreeModel*, MaxQSLeavesCount * 2> curLevel; // current level nodes
	curLevel.Add( root );

	// Traverse the tree breadth first until we get MaxQSLeavesCount leaves
	while( !curLevel.IsEmpty() ) {
		for( int i = curLevel.Size() - 1; i >= 0; i-- ) {
			const IRegressionTreeModel* node = curLevel[i];

			CRegressionTreeNodeInfo nodeInfo;
			node->GetNodeInfo(nodeInfo);
			if( nodeInfo.Type == RTNT_Const ) {
				qsLeaves.Add( node );
				curLevel.DeleteAt( i );
			} else if( nodeInfo.Type == RTNT_Continuous ) {
				nextLevel.Add( node->GetLeftChild() );
				nextLevel.Add( node->GetRightChild() );
			} else {
				NeoAssert( false );
			}
		}

		if( nextLevel.Size() + qsLeaves.Size() > MaxQSLeavesCount ) {
			// The whole next level does not fit in
			for( int i = 0; i < curLevel.Size(); i++ ) {
				qsLeaves.Add( curLevel[i] );
			}
			break;
		}
		nextLevel.MoveTo( curLevel );
	}

	// All nodes on this level will not fit in, put in as many as possible
	for( int i = 0; i < curLevel.Size() && qsLeaves.Size() < MaxQSLeavesCount; i++ ) {
		const IRegressionTreeModel* node = curLevel[i];

		CRegressionTreeNodeInfo nodeInfo;
		node->GetNodeInfo(nodeInfo);

		if( nodeInfo.Type == RTNT_Continuous ) {
			qsLeaves.Delete( node );
			qsLeaves.Add( node->GetLeftChild() );
			qsLeaves.Add( node->GetRightChild() );
		}
	}
}

// Serializes an integer depending on its value (like UTF).
inline static void serializeCompact( CArchive& archive, unsigned int& value )
{
	const int UsefulBitCount = 7;
	const int UsefulBitMask = ( 1 << UsefulBitCount ) - 1;
	const int ContinueBitMask = ( 1 << UsefulBitCount );

	if( archive.IsStoring() ) {
		unsigned int temp = value;
		do {
			unsigned int div = temp >> UsefulBitCount;
			unsigned int mod = temp & UsefulBitMask;
			archive << static_cast<unsigned char>( mod | ( div > 0 ? ContinueBitMask : 0 ) );
			temp = div;
		} while( temp > 0 );
	} else if( archive.IsLoading() ) {
		value = 0;
		unsigned int shift = 0;
		unsigned char mod = 0;
		do {
			archive >> mod;
			value = ( ( mod & UsefulBitMask ) << shift ) + value;
			shift += UsefulBitCount;
		} while( mod & ContinueBitMask );
	} else {
		NeoAssert( false );
	}
}

//------------------------------------------------------------------------------------------------------------

// Serialization interface for the optimized tree
class IQsSerializer {
public:
	virtual ~IQsSerializer() {}
	
	// Reads a node
	virtual void Read( int& featureIndex, float& value, bool& isQsLeaf ) = 0;
	// Writes a node
	virtual void Write( int featureIndex, float value, bool isQsLeaf ) = 0;
};

// Serializes a tree into an archive
class CArchiveQsSerializer : public IQsSerializer {
public:
	CArchiveQsSerializer( CArchive& _archive, bool _noSimpleNodes ) : archive( _archive ), noSimpleNodes( _noSimpleNodes ) {}

	// IQsSerializer interface methods
	void Read( int& featureIndex, float& value, bool& isQsLeaf ) override;
	void Write( int featureIndex, float value, bool isQsLeaf ) override;

private:
	CArchive& archive; // the archive that is used
	const int noSimpleNodes; // indicates if there are non-optimized nodes
};

void CArchiveQsSerializer::Read( int& featureIndex, float& value, bool& isQsLeaf )
{
	unsigned int rawFeatureIndex = 0;
	serializeCompact( archive, rawFeatureIndex );
	archive >> value;
	
	if( noSimpleNodes ) {
		featureIndex = rawFeatureIndex == 0 ? NotFound : ( rawFeatureIndex - 1 );
		isQsLeaf = ( featureIndex == NotFound );
	} else {
		isQsLeaf = ( rawFeatureIndex % 2 == 1 );
		featureIndex = rawFeatureIndex / 2 == 0 ? NotFound : static_cast<unsigned int>( rawFeatureIndex / 2 - 1 );
	}
}

void CArchiveQsSerializer::Write( int featureIndex, float value, bool isQsLeaf )
{
	unsigned int rawFeatureIndex = 0;
	if( noSimpleNodes ) {
		rawFeatureIndex = featureIndex == NotFound ? 0 : static_cast<unsigned int>( featureIndex + 1 );
	} else {
		// We need an indicator of the node type; use the lowest bit
		rawFeatureIndex = featureIndex == NotFound ? 0 : static_cast<unsigned int>( ( featureIndex + 1 ) * 2 );
		if( isQsLeaf ) {
			rawFeatureIndex++;
		}
	}
	// There are generally far fewer features than 2^31, so there numbers may be stored more efficiently
	serializeCompact( archive, rawFeatureIndex );
	archive << value;
}

//------------------------------------------------------------------------------------------------------------

// Serializes the gradient boosting tree
class CGBEnsembleQsSerializer : public IQsSerializer {
public:
	CGBEnsembleQsSerializer( const IRegressionTreeModel* root, const CHashTable<const IRegressionTreeModel*>& qsLeaves );

	// IQsSerializer interface methods
	void Read( int& featureIndex, float& value, bool& isQsLeaf ) override;
	void Write( int, float, bool ) override { NeoAssert( false ); } // we only need to read from a tree

private:
	static const unsigned int S_NodeProcessed = 1;
	static const unsigned int S_LeftProcessed = 2;
	static const unsigned int S_RightProcessed = 4;
	static const unsigned int S_QSNode = 8;

	struct CStackNode {
		const IRegressionTreeModel* Node;
		DWORD State;

		explicit CStackNode( const IRegressionTreeModel* node, DWORD state ) : Node( node ), State( state ) { NeoAssert( node != 0 ); }
	};

	const CHashTable<const IRegressionTreeModel*>& qsLeaves; // optimized leaves hash

	CFastArray<CStackNode, 32> stack; // the stack for depth-first search
};

CGBEnsembleQsSerializer::CGBEnsembleQsSerializer( const IRegressionTreeModel* root, const CHashTable<const IRegressionTreeModel*>& _qsLeaves ) :
	qsLeaves( _qsLeaves )
{
	DWORD state = !qsLeaves.Has( root ) ? S_QSNode : 0;
	stack.Add( CStackNode( root, state ) );
}

void CGBEnsembleQsSerializer::Read( int& featureIndex, float& value, bool& isQsLeaf )
{
	// Read the nodes depth first
	NeoAssert( !stack.IsEmpty() );

	CRegressionTreeNodeInfo info;
	const IRegressionTreeModel* result = 0;

	while( !stack.IsEmpty() ) {
		const IRegressionTreeModel* node = stack.Last().Node;
		DWORD& state = stack.Last().State;

		node->GetNodeInfo( info );

		if( !HasFlag( state, S_NodeProcessed ) ) { // has this node been read?
			SetFlags( state, S_NodeProcessed );
			result = node;
			break;
		}

		// For the non-optimized and non-inverted nodes, read left child first
		// For the optimized inverted nodes, read right child first
		if( HasFlag( state, S_QSNode ) && info.Value < 0 ) {
			if( !HasFlag( state, S_RightProcessed ) && info.Type != RTNT_Const ) { // has the right subtree been read?
				SetFlags( state, S_RightProcessed );
				DWORD childState = qsLeaves.Has( node->GetRightChild() ) ? 0 : S_QSNode;
				stack.Add( CStackNode( node->GetRightChild(), childState ) );
				continue;
			}

			if( !HasFlag( state, S_LeftProcessed ) && info.Type != RTNT_Const ) { // has the left subtree been read?
				SetFlags( state, S_LeftProcessed );
				DWORD childState = qsLeaves.Has( node->GetLeftChild() ) ? 0 : S_QSNode;
				stack.Add( CStackNode( node->GetLeftChild(), childState ) );
				continue;
			}
		} else {
			if( !HasFlag( state, S_LeftProcessed ) && info.Type != RTNT_Const ) { // has the left subtree been read?
				SetFlags( state, S_LeftProcessed );
				DWORD childState = ( HasFlag( state, S_QSNode ) && !qsLeaves.Has( node->GetLeftChild() ) ) ? S_QSNode : 0;
				stack.Add( CStackNode( node->GetLeftChild(), childState ) );
				continue;
			}

			if( !HasFlag( state, S_RightProcessed ) && info.Type != RTNT_Const ) { // has the right subtree been read?
				SetFlags( state, S_RightProcessed );
				DWORD childState = ( HasFlag( state, S_QSNode ) && !qsLeaves.Has( node->GetRightChild() ) ) ? S_QSNode : 0;
				stack.Add( CStackNode( node->GetRightChild(), childState ) );
				continue;
			}
		}

		stack.DeleteLast();
	}

	NeoAssert( result != 0 );
	// Get the current node data
	featureIndex = info.FeatureIndex;
	value = static_cast<float>( info.Value );
	isQsLeaf = qsLeaves.Has( result );
}

//------------------------------------------------------------------------------------------------------------

void CGradientBoostQSEnsemble::Build( const CGradientBoostEnsemble &treeModel )
{
	NeoAssert( treeModel.Size() <= MaxTreesCount );

	const int treeCount = treeModel.Size();
	treeQsLeavesOffsets.SetSize( treeCount );

	// Feature indices will be compressed for compact representation
	CArray<int> features; // the "optimized node -> feature index" mapping
	CHashTable<const IRegressionTreeModel*> qsLeavesTable; // the table of optimized subtree leaves
	// For all optimized subtrees fill in lessNodes and moreNodes
	for( int i = 0; i < treeCount; i++ ) {
		treeQsLeavesOffsets[i] = qsLeaves.Size();

		CPtr<const IRegressionTreeModel> tree = CheckCast<const IRegressionTreeModel>( treeModel[i] );
		// Choose an optimized subtree and write the leaves into the table
		findQsLeaves( tree, qsLeavesTable );

		// Build the internal representation of the tree
		CGBEnsembleQsSerializer serializer( tree, qsLeavesTable );
		int startOrder = 0;
		bool isQsLeaf = false;
		unsigned __int64 mask = 0;
		loadQSNode( serializer, i, startOrder, isQsLeaf, mask, features );

		qsLeavesTable.DeleteAll(); // the table is only for one tree
		features.Add( NotFound ); // we need the treeQsLeavesOffsets[i] to point to features as well (while there are 1 more leaves than features)
	}

	// Find the offsets for nodes of each feature
	buildFeatureNodesOffsets( features );
}

double CGradientBoostQSEnsemble::Predict( const CSparseFloatVector& data ) const
{
	// The resulting bit masks, one per tree; for a start all bits are set to 1
	CFastArray<unsigned __int64, 512> resultBitvectors;
	resultBitvectors.SetSize( GetTreesCount() );
	memset( resultBitvectors.GetPtr(), ~0, resultBitvectors.Size() * sizeof( unsigned __int64 ) );

	const CSparseFloatVectorDesc& desc = data.GetDesc();
	for( int i = 0; i < desc.Size; i++ ) {
		processFeature( desc.Indexes[i], desc.Values[i], resultBitvectors );
	}

	return calculateScore<CSparseFloatVector>( data, resultBitvectors, GetTreesCount() - 1 );
}

double CGradientBoostQSEnsemble::Predict( const CFloatVector& data ) const
{
	CFastArray<unsigned __int64, 512> resultBitvectors;
	resultBitvectors.SetSize( GetTreesCount() );
	memset( resultBitvectors.GetPtr(), ~0, resultBitvectors.Size() * sizeof( unsigned __int64 ) );

	for( int i = 0; i < data.Size(); i++ ) { 
		processFeature( i, data[i], resultBitvectors );
	}

	return calculateScore<CFloatVector>( data, resultBitvectors, GetTreesCount() - 1 );
}

double CGradientBoostQSEnsemble::Predict( const CSparseFloatVectorDesc& data ) const
{
	// The resulting bit masks, one per tree; for a start all bits are set to 1
	CFastArray<unsigned __int64, 512> resultBitvectors;
	resultBitvectors.SetSize( GetTreesCount() );
	memset( resultBitvectors.GetPtr(), ~0, resultBitvectors.Size() * sizeof( unsigned __int64 ) );

	for( int i = 0; i < data.Size; i++ ) {
		processFeature( data.Indexes[i], data.Values[i], resultBitvectors );
	}

	return calculateScore<CSparseFloatVectorDesc>( data, resultBitvectors, GetTreesCount() - 1 );
}

double CGradientBoostQSEnsemble::Predict( const CSparseFloatVectorDesc& data, int lastTreeIndex ) const
{
	CFastArray<unsigned __int64, 512> resultBitvectors;
	resultBitvectors.SetSize( GetTreesCount() );
	memset( resultBitvectors.GetPtr(), ~0, resultBitvectors.Size() * sizeof( unsigned __int64 ) );

	for( int i = 0; i < data.Size; i++ ) {
		processFeature( data.Indexes[i], data.Values[i], resultBitvectors );
	}

	return calculateScore<CSparseFloatVectorDesc>( data, resultBitvectors, lastTreeIndex );
}

CArchive& operator<<( CArchive& archive, const CGradientBoostQSEnsemble& block )
{
	archive.SerializeVersion( 0 );
	block.store( archive );
	return archive;
}

CArchive& operator>>( CArchive& archive, CGradientBoostQSEnsemble& block )
{
	archive.SerializeVersion( 0 );
	block.load( archive );
	return archive;
}

// The comparer that sorts the references to optimized nodes
// Used for generating the depth-first traverse order
class CQSNodeLinkAscending {
public:
	explicit CQSNodeLinkAscending( const CArray<CQSNode>& _nodes ) : nodes( _nodes ) {}

	bool Predicate( const int& first, const int& second ) const;
	bool IsEqual( const int& first, const int& second ) const;
	void Swap( int& first, int& second ) const { swap<int>( first, second ); }

private:
	const CArray<CQSNode>& nodes;
};

bool CQSNodeLinkAscending::Predicate( const int& first, const int& second ) const
{
	if( nodes[first].Tree < nodes[second].Tree ) {
		return true;
	} else if( nodes[first].Tree > nodes[second].Tree ) {
		return false;
	} else {
		return ( nodes[first].Order < nodes[second].Order );
	}
}

bool CQSNodeLinkAscending::IsEqual( const int& first, const int& second ) const
{
	return ( nodes[first].Tree == nodes[second].Tree ) && ( nodes[first].Order == nodes[second].Order );
}

// Writes an ensemble of trees into archive
void CGradientBoostQSEnsemble::store( CArchive& archive ) const
{
	// The trees will be stored in depth-first traversal order, using utf8size(featureIndex) + size(threshold) bytes per node
	archive << qsNodes.Size();
	archive << featureQsNodesOffsets.Size();
	archive << qsLeaves.Size();
	archive << treeQsLeavesOffsets.Size();
	archive << simpleNodes.Size();

	CArray<int> features;
	buildNodesFeatures( features );

	CArray<int> links;
	links.SetBufferSize( qsNodes.Size() );

	for( int i = 0; i < qsNodes.Size(); i++ ) {
		links.Add( i );
	}

	// Sort the optimized subtree nodes in depth-first traversal order
	CQSNodeLinkAscending comparator( qsNodes );
	links.QuickSort<CQSNodeLinkAscending>( &comparator );

	CArchiveQsSerializer serializer( archive, simpleNodes.IsEmpty() );

	int leafIndex = 0;
	int nodeIndex = 0;
	for( int i = 0; i < treeQsLeavesOffsets.Size(); i++ ) {
		if( nodeIndex < links.Size() && i == qsNodes[links[nodeIndex]].Tree ) {
			storeQSNode( serializer, links, features, leafIndex, nodeIndex );
		} else {
			// The tree number i is empty
			storeQSLeaf( serializer, leafIndex );
		}
	}
}

// Writes a node of the optimized subtree into archive
void CGradientBoostQSEnsemble::storeQSNode( IQsSerializer& serializer, const CArray<int>& links,
	const CArray<int>& features, int& leafIndex, int& nodeIndex ) const
{
	const CQSNode& curNode = qsNodes[links[nodeIndex]];

	serializer.Write( features[links[nodeIndex]], curNode.Threshold, false );
	nodeIndex++;

	// If the node has been inverted, the right child goes first, the left one second
	if( curNode.Threshold < 0 ) {
		// Process the right child
		if( HasFlag( curNode.PropertiesMask, PM_RightLeaf ) ) {
			storeQSLeaf( serializer, leafIndex );
		} else {
			storeQSNode( serializer, links, features, leafIndex, nodeIndex );
		}

		// Process the left child
		if( HasFlag( curNode.PropertiesMask, PM_LeftLeaf ) ) {
			storeQSLeaf( serializer, leafIndex );
		} else {
			storeQSNode( serializer, links, features, leafIndex, nodeIndex );
		}
	} else {
		// Process the left child
		if( HasFlag( curNode.PropertiesMask, PM_LeftLeaf ) ) {
			storeQSLeaf( serializer, leafIndex );
		} else {
			storeQSNode( serializer, links, features, leafIndex, nodeIndex );
		}

		// Process the right child
		if( HasFlag( curNode.PropertiesMask, PM_RightLeaf ) ) {
			storeQSLeaf( serializer, leafIndex );
		} else {
			storeQSNode( serializer, links, features, leafIndex, nodeIndex );
		}
	}
}

// Writes into archive the branch of the original tree that starts in the given leaf of the optimized subtree
void CGradientBoostQSEnsemble::storeQSLeaf( IQsSerializer& serializer, int& leafIndex ) const
{
	const CQSLeaf& leaf = qsLeaves[leafIndex];
	leafIndex++;
	if( leaf.SimpleNodeIndex == NotFound ) {
		serializer.Write( NotFound, leaf.Value, true );
		return;
	}

	storeSimpleNode( serializer, leaf.SimpleNodeIndex );
}

// Writes into archive the branch that starts in the given node of the original non-optimized tree
void CGradientBoostQSEnsemble::storeSimpleNode( IQsSerializer& serializer, int index ) const
{
	const CSimpleNode& node = simpleNodes[index];
	const int featureIndex = node.Feature;
	if( featureIndex == NotFound ) {
		serializer.Write( featureIndex, node.Value, true );
		return;
	}

	serializer.Write( featureIndex, node.Value, true );

	storeSimpleNode( serializer, index + 1 );
	storeSimpleNode( serializer, node.RightChild );
}

// Builds the array of features used in the optimized nodes
void CGradientBoostQSEnsemble::buildNodesFeatures( CArray<int>& features ) const
{
	features.Empty();

	features.Add( NotFound, qsNodes.Size() );
	for( int i = featureQsNodesOffsets.GetFirstPosition(); i != NotFound; i = featureQsNodesOffsets.GetNextPosition( i ) ) {
		const int featureIndex = featureQsNodesOffsets.GetKey( i );
		const CQSNodeOffset& offset = featureQsNodesOffsets.GetValue( i );
		if( offset.Less.Begin != NotFound ) {
			for( int j = offset.Less.Begin; j <= offset.Less.End; j++ ) {
				features[j] = featureIndex;
			}
		}
;
		if( offset.More.Begin != NotFound ) {
			for( int j = offset.More.Begin; j <= offset.More.End; j++ ) {
				features[j] = featureIndex;
			}
		}
	}
}

// Reads a tree ensemble from archive
void CGradientBoostQSEnsemble::load( CArchive& archive )
{
	int qsNodesSize = 0;
	archive >> qsNodesSize;
	qsNodes.SetBufferSize( qsNodesSize );

	int featureQsNodesOffsetsSize = 0;
	archive >> featureQsNodesOffsetsSize;
	featureQsNodesOffsets.SetHashTableSize( featureQsNodesOffsetsSize );

	int qsLeavesSize = 0;
	archive >> qsLeavesSize;
	qsLeaves.SetBufferSize( qsLeavesSize );

	int treeQsLeavesOffsetsSize = 0;
	archive >> treeQsLeavesOffsetsSize;
	treeQsLeavesOffsets.SetSize( treeQsLeavesOffsetsSize );

	int simpleNodesSize = 0;
	archive >> simpleNodesSize;
	simpleNodes.SetBufferSize( simpleNodesSize );

	CArchiveQsSerializer serializer( archive, simpleNodesSize == 0 );

	// Feature indices can be compressed for more compact representation
	CArray<int> features; // the "algorithm feature index --> original feature index"

	for( int i = 0; i < treeQsLeavesOffsets.Size(); i++ ) {
		treeQsLeavesOffsets[i] = qsLeaves.Size();

		int startOrder = 0;
		bool isQsLeaf = false;
		unsigned __int64 mask = 0;
		loadQSNode( serializer, i, startOrder, isQsLeaf, mask, features );
		features.Add( NotFound ); // so that treeQsLeavesOffsets[i] points to features as well (while there are one more leaves than features)
	}

	// Find the offsets for each feature nodes
	buildFeatureNodesOffsets( features );
}

// Reads from archive a node in the optimized subtree
void CGradientBoostQSEnsemble::loadQSNode( IQsSerializer& serializer, int tree,
	int& order, bool& isQsLeaf, unsigned __int64& mask, CArray<int>& features )
{
	int featureIndex = NotFound;
	float threshold = 0;
	isQsLeaf = false;
	serializer.Read( featureIndex, threshold, isQsLeaf );

	mask = 0;
	if( !isQsLeaf ) {
		// The current node is to be optimized
		qsNodes.Add( CQSNode( 0, threshold, tree, order, 0 ) );
		const int index = qsNodes.Size() - 1;
		features.Add( featureIndex );
		order++;

		unsigned __int64 leftMask = 0;
		bool leftIsQsLeaf = false;
		unsigned __int64 rightMask = 0;
		bool rightIsQsLeaf = false;
		if( threshold < 0 ) {
			loadQSNode( serializer, tree, order, rightIsQsLeaf, rightMask, features );
			loadQSNode( serializer, tree, order, leftIsQsLeaf, leftMask, features );
		} else {
			loadQSNode( serializer, tree, order, leftIsQsLeaf, leftMask, features );
			loadQSNode( serializer, tree, order, rightIsQsLeaf, rightMask, features );
		}
		mask = leftMask & rightMask; 

		// The zero values should be in the left subtree, otherwise invert the node
		unsigned char propertiesMask = 0;
		propertiesMask |= ( threshold < 0 ? PM_Inverted : 0 );
		propertiesMask |= ( leftIsQsLeaf ? PM_LeftLeaf : 0 );
		propertiesMask |= ( rightIsQsLeaf ? PM_RightLeaf : 0 );
		qsNodes[index].Mask = threshold < 0 ? rightMask : leftMask;
		qsNodes[index].PropertiesMask = propertiesMask;
	} else {
		// The leaf node
		// The leaves are numbered left to right; when computing the score find the lowest nonzero bit index
		const int leafIndex = qsLeaves.Size() - treeQsLeavesOffsets[tree];
		mask = ~( static_cast<unsigned __int64>(1) << leafIndex );

		loadQSLeaf( serializer, featureIndex, threshold );
	}
}

// Reads from archive a leaf in the optimized subtree
void CGradientBoostQSEnsemble::loadQSLeaf( IQsSerializer& serializer, int featureIndex, float threshold )
{
	if( featureIndex == NotFound ) {
		qsLeaves.Add( CQSLeaf( static_cast<float>(threshold) ) );
		return;
	}
	
	qsLeaves.Add( CQSLeaf( simpleNodes.Size() ) );
	loadSimpleSubtree( serializer, featureIndex, threshold );
}

// Reads from archive a branch of the original tree
void CGradientBoostQSEnsemble::loadSimpleSubtree( IQsSerializer& serializer, int featureIndex, float threshold )
{
	if( featureIndex == NotFound ) {
		simpleNodes.Add( CSimpleNode( static_cast<float>(threshold) ) );
		return;
	}

	const int currentNodeIndex = simpleNodes.Size();
	simpleNodes.Add( CSimpleNode( featureIndex, threshold, NotFound ) );

	bool isQsNode = false;
	serializer.Read( featureIndex, threshold, isQsNode );
	loadSimpleSubtree( serializer, featureIndex, threshold );

	simpleNodes[currentNodeIndex].RightChild = simpleNodes.Size();

	serializer.Read( featureIndex, threshold, isQsNode );
	loadSimpleSubtree( serializer, featureIndex, threshold );
}

// The comparer that sorts the references to optimized nodes
// Used for generating the order optimal for classification
class CQSNodeAscending {
public:
	CQSNodeAscending( const CArray<int>& _features, const CArray<int>& _treeOffsets );

	bool Predicate( const CQSNode& first, const CQSNode& second ) const;
	bool IsEqual( const CQSNode& first, const CQSNode& second ) const;
	void Swap( CQSNode& first, CQSNode& second ) const { swap<CQSNode>( first, second ); }

private:
	const CArray<int>& features;
	const CArray<int>& treeOffsets;
};

inline CQSNodeAscending::CQSNodeAscending( const CArray<int>& _features, const CArray<int>& _treeOffsets ) :
	features( _features ),
	treeOffsets( _treeOffsets )
{
}

inline bool CQSNodeAscending::Predicate( const CQSNode& first, const CQSNode& second ) const
{
	const bool firstIsInverted = HasFlag( first.PropertiesMask, PM_Inverted );
	const bool secondIsInverted = HasFlag( second.PropertiesMask, PM_Inverted );
	if( firstIsInverted != secondIsInverted ) {
		return ( firstIsInverted < secondIsInverted );
	}

	const int firstTreeOrderIndex = treeOffsets[first.Tree] + first.Order;
	const int secondTreeOrderIndex = treeOffsets[second.Tree] + second.Order;
	if( features[firstTreeOrderIndex] != features[secondTreeOrderIndex] ) {
		return features[firstTreeOrderIndex] < features[secondTreeOrderIndex];
	}

	if( firstIsInverted == 0 ) {
		return ( first.Threshold < second.Threshold );
	}
	return ( second.Threshold < first.Threshold );
}

inline bool CQSNodeAscending::IsEqual( const CQSNode& first, const CQSNode& second ) const
{
	return ( HasFlag( first.PropertiesMask, PM_Inverted ) == HasFlag( second.PropertiesMask, PM_Inverted ) )
		&& ( features[treeOffsets[first.Tree] + first.Order] == features[treeOffsets[second.Tree] + second.Order] )
		&& ( first.Threshold == second.Threshold );
}

// Finds the range for each feature for subsequent better search
void CGradientBoostQSEnsemble::buildFeatureNodesOffsets( const CArray<int>& features )
{
	// Sort the non-leaf nodes by features and split thresholds
	CQSNodeAscending comparator( features, treeQsLeavesOffsets );
	qsNodes.QuickSort<CQSNodeAscending>( &comparator );

	featureQsNodesOffsets.Empty();
	for( int i = 0; i < qsNodes.Size(); i++ ) {
		const int featureIndex = features[treeQsLeavesOffsets[qsNodes[i].Tree] + qsNodes[i].Order];
		CQSNodeOffset& offset = featureQsNodesOffsets.GetOrCreateValue( featureIndex );
		if( HasFlag( qsNodes[i].PropertiesMask, PM_Inverted ) ) {
			if( offset.More.Begin == NotFound ) {
				offset.More.Begin = i;
			}
			offset.More.End = i;
		} else {
			if( offset.Less.Begin == NotFound ) {
				offset.Less.Begin = i;
			}
			offset.Less.End = i;
		}
	}
}

// Mask computation
// From the start, all bitvectors elements are filled with ones. 
// Traverse all nodes that use the given feature; if the condition is not fulfilled, 
// calculate bitwise AND of the current bitvector with the node mask. 
// Once the condition is fulfilled, stop because all the rest will be fulfilled also.
void CGradientBoostQSEnsemble::processFeature( int featureIndex, float value, CFastArray<unsigned __int64, 512>& bitvectors ) const
{
	CQSNodeOffset offset;
	if( !featureQsNodesOffsets.Lookup( featureIndex, offset ) ) {
		return;
	}

	if( offset.Less.Begin != NotFound ) {
		for( int i = offset.Less.Begin; i <= offset.Less.End && qsNodes[i].Threshold < value; i++ ) {
			const CQSNode& node = qsNodes[i];
			bitvectors[node.Tree] &= node.Mask;
		}
	}

	if( offset.More.Begin != NotFound ) {
		for( int i = offset.More.Begin; i <= offset.More.End && qsNodes[i].Threshold >= value; i++ ) {
			const CQSNode& node = qsNodes[i];
			bitvectors[node.Tree] &= node.Mask;
		}
	}
}

static inline float getFeatureValue( const CSparseFloatVector& data, int index )
{
	return data.GetValue( index );
}

static inline float getFeatureValue( const CFloatVector& data, int index )
{
	return data[index];
}

static inline float getFeatureValue( const CSparseFloatVectorDesc& data, int index )
{
	float result;
	GetValue( data, index, result );
	return result;
}

// Score computation
// The leaves are numbered left to right (all masks are inverted), so look for the lowest nonzero bit
// In each bitvector the leaf we need has the index of the lowest nonzero
// If it is a leaf in the original tree, take its value, if a subtree call its Predict method
template <class T>
double CGradientBoostQSEnsemble::calculateScore( const T& data, const CFastArray<unsigned __int64, 512>& bitvectors, int lastTreeIndex ) const
{
	float score = 0.0;
	int prev = -1;
	const int end = min( lastTreeIndex, GetTreesCount() - 1 );
	for( int i = 0; i <= end; i++ ) {
		const int leafIndex = findLowestBitIndex( bitvectors[i] );
		const int currentTreeOffset = treeQsLeavesOffsets[i];
		NeoAssert( prev != currentTreeOffset );
		prev = currentTreeOffset;
		const CQSLeaf& leaf = qsLeaves[currentTreeOffset + leafIndex];

		if( leaf.SimpleNodeIndex == NotFound ) {
			score += leaf.Value;
		} else {
			int nodeIndex = leaf.SimpleNodeIndex;

			while( simpleNodes[nodeIndex].Feature != NotFound ) {
				if( getFeatureValue( data, simpleNodes[nodeIndex].Feature ) <= simpleNodes[nodeIndex].Value ) {
					nodeIndex++;
				} else {
					nodeIndex = simpleNodes[nodeIndex].RightChild;
				}
			}
			score += simpleNodes[nodeIndex].Value;
		}
	}
	return score;
}

} // namespace NeoML
