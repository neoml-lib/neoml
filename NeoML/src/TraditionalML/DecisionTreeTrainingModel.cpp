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

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/DecisionTreeTrainingModel.h>
#include <DecisionTreeNodeBase.h>
#include <DecisionTreeClassificationModel.h>
#include <DecisionTreeNodeClassificationStatistic.h>
#include <float.h>

namespace NeoML {

IDecisionTreeModel::~IDecisionTreeModel()
{
}

//---------------------------------------------------------------------------------------------------------

const int CDecisionTreeTrainingModel::MaxClassifyNodesCacheSize;

CDecisionTreeTrainingModel::CDecisionTreeTrainingModel( const CParams& _params, CRandom* _random ) :
	params( _params ),
	random( _random ),
	logStream( 0 ),
	nodesCount( 0 ),
	statisticsCacheSize( 0 )
{
	NeoAssert( params.MinDiscreteSubsetSize > 0 );
	NeoAssert( params.MinContinuousSubsetSize > 0 );
	NeoAssert( params.MinSplitSize > 0 );
	NeoAssert( params.MinContinuousSubsetPart >= 0 );
	NeoAssert( params.MinContinuousSubsetPart <= 1 );
	NeoAssert( params.MinDiscreteSubsetPart >= 0 );
	NeoAssert( params.MinDiscreteSubsetPart <= 1 );
	NeoAssert( params.MaxTreeDepth > 0 );
	NeoAssert( params.MaxNodesCount > 1 );
	NeoAssert( 0.00 <= params.ConstNodeThreshold && params.ConstNodeThreshold <= 1.0 );
}

CDecisionTreeTrainingModel::~CDecisionTreeTrainingModel()
{
}

CPtr<IModel> CDecisionTreeTrainingModel::Train( const IProblem& problem )
{
	NeoAssert( problem.GetVectorCount() > 0 );
	NeoAssert( problem.GetClassCount() > 0 );
	NeoAssert( problem.GetFeatureCount() > 0 );

	classificationProblem = &problem;
	CPtr<CDecisionTreeClassificationModel> root =
		dynamic_cast<CDecisionTreeClassificationModel*>( buildTree( problem.GetVectorCount() ).Ptr() );

	return root.Ptr();
}

CPtr<CDecisionTreeNodeBase> CDecisionTreeTrainingModel::buildTree( int vectorCount )
{
	if( logStream != 0 ) {
		*logStream << "\nDecision tree training started:\n";
	}

	// Create the tree root and gather statistics for it
	// Based on this data, decide what size cache we will need and can afford
	CPtr<CDecisionTreeNodeBase> root = createNode();
	nodesCount = 1;
	CDecisionTreeNodeStatisticBase* rootStatistic = createStatistic( root );
	CSparseFloatMatrixDesc matrix = classificationProblem->GetMatrix();

	CSparseFloatVectorDesc desc;
	for( int i = 0; i < vectorCount; i++ ) {
		matrix.GetRow( i, desc );
		rootStatistic->AddVector( i, desc );
	}
	rootStatistic->Finish();

	classifyNodesCache.Empty();
	classifyNodesLevel.Empty();
	const int classifyNodesCacheSize = min( MaxClassifyNodesCacheSize, vectorCount );
	classifyNodesCache.Add( root, classifyNodesCacheSize );
	classifyNodesLevel.Add( 0, classifyNodesCacheSize );

	statisticsCacheSize = static_cast<int>( params.AvailableMemory / rootStatistic->GetSize() );
	NeoAssert( statisticsCacheSize > 0 ); // we need at least the amount of memory sufficient for one node statistics
	statisticsCache.FreeBuffer();
	statisticsCache.SetBufferSize( statisticsCacheSize );

	split( *rootStatistic, 0 );
	delete rootStatistic;

	// Build the tree level by level
	for( int i = 1; i <= params.MaxTreeDepth; i++ ) {
		if( !buildTreeLevel( matrix, i, *root ) ) {
			break;
		}
	}

	statisticsCache.FreeBuffer();

	if( logStream != 0 ) {
		*logStream << "\nDecision tree training finished\n";
	}

	return root;
}

// Builds one tree level
bool CDecisionTreeTrainingModel::buildTreeLevel( const CSparseFloatMatrixDesc& matrix, int level, CDecisionTreeNodeBase& root ) const
{
	if( logStream != 0 ) {
		*logStream << "\nBuild level " << level << ":\n";
	}

	// We can keep the information only about statisticsCache.Size() nodes in memory at the same time
	// If there are more nodes at this level we will need to make several passes

	bool allNodesProccessed = false;
	int step = 0;
	bool result = false;

	while( !allNodesProccessed ) { // until all nodes on the level are processed

		if( logStream != 0 ) {
			*logStream << "\nBuild level " << level << " step " << step << ":\n";
		}

		statisticsCache.Empty();

		// Gather statistics
		allNodesProccessed = collectStatistics( matrix, level, &root );

		if( logStream != 0 ) {
			if( allNodesProccessed ) {
				*logStream << "\nStatistics collected for all nodes.\n";
			} else {
				*logStream << "\nStatistics collected partially.\n";
			}
		}

		// Split according to the statistics just gathered
		for( int i = 0; i < statisticsCache.Size(); i++ ) {
			if( split( *statisticsCache[i], level ) ) {
				result = true;
			}
		}

		step++;
	}
	return result;
}

// Gathers statistics for the nodes of one level
// Returns true if all nodes were traversed and false if another pass is needed
bool CDecisionTreeTrainingModel::collectStatistics( const CSparseFloatMatrixDesc& matrix, int level, CDecisionTreeNodeBase* root ) const
{
	NeoAssert( level > 0 );
	NeoAssert( root != 0 );
	CMap<CDecisionTreeNodeBase*, int> nodesStatistics;

	bool result = true;
	const int matrixHeight = matrix.Height;

	for( int i = 0; i < matrixHeight; i++ ) {
		CSparseFloatVectorDesc vector;
		matrix.GetRow( i, vector );
		// Find the leaf node for this vector in the current tree
		CPtr<CDecisionTreeNodeBase> leaf;
		int leafLevel = 0;
		if( i < MaxClassifyNodesCacheSize ) {
			classifyNodesCache[i]->GetClassifyNode( vector, leaf, leafLevel );
			leafLevel += classifyNodesLevel[i];
			classifyNodesCache[i] = leaf;
			classifyNodesLevel[i] = leafLevel;
		} else {
			root->GetClassifyNode( vector, leaf, leafLevel );
		}

		if( leafLevel != level || leaf->GetType() != DTNT_Undefined ) {
			// This node belongs to another level or was already processed on the current level
			continue;
		}

		TMapPosition pos = nodesStatistics.GetFirstPosition( leaf );
		int nodeStatisticIndex = NotFound;
		if( pos == NotFound ) {
			const int curStatisticsCashSize = nodesStatistics.Size();
			// No statistics object for this node, create one
			if( curStatisticsCashSize >= statisticsCacheSize ) {
				// Insufficient space in cache; this node will be processed on another pass
				result = false;
				continue;
			}
			nodeStatisticIndex = curStatisticsCashSize;
			statisticsCache.Add( createStatistic( leaf ) );
			nodesStatistics.Add( leaf, nodeStatisticIndex );
		} else {
			nodeStatisticIndex = nodesStatistics.GetValue( pos );
		}

		statisticsCache[nodeStatisticIndex]->AddVector( i, vector );
	}

	for( int i = 0; i < statisticsCache.Size(); i++ ) {
		statisticsCache[i]->Finish();
	}

	return result;
}

// Splits the specified node according to the accumulated statistics
// Returns true if new nodes were created when splitting
bool CDecisionTreeTrainingModel::split( const CDecisionTreeNodeStatisticBase& nodeStatistics, int level ) const
{
	CDecisionTreeNodeBase& node = nodeStatistics.GetNode();
	CArray<double> predictions;
	const double maxProbability = nodeStatistics.GetPredictions( predictions );

	if( logStream != 0 ) {
		*logStream << "\nSplit node contains " << nodeStatistics.GetVectorsCount() << " vectors.\n";
		for( int i = 0; i < predictions.Size(); i++ ) {
			*logStream << "Class " << i << ": prediction = " << predictions[i] << " \n";
		}
	}

	// Create a constant node for too similar or too small sets
	if( ( predictions.Size() > 1 && maxProbability >= params.ConstNodeThreshold )
		|| nodeStatistics.GetVectorsCount() < params.MinSplitSize )
	{
		if( logStream != 0 ) {
			*logStream << "Split result: created const node.\n";
		}

		CDecisionTreeConstNodeInfo* info  = FINE_DEBUG_NEW CDecisionTreeConstNodeInfo();
		predictions.MoveTo( info->Predictions );
		node.SetInfo( info );
		return false;
	}

	bool isBestFeatureDiscrete = false;
	int bestFeature = NotFound;
	CArray<double> bestSplitValues;
	double bestCriterionValue = DBL_MAX;

	if( nodeStatistics.GetSplit( params, isBestFeatureDiscrete, bestFeature, bestSplitValues, bestCriterionValue )
		&& nodesCount + bestSplitValues.Size() <= params.MaxNodesCount
		&& level < params.MaxTreeDepth )
	{
		// The new node is NOT a leaf

		if( logStream != 0 ) {
			*logStream << "Split result: splited by feature: " << bestFeature << " value = " << bestCriterionValue << "\n";
		}

		nodesCount += bestSplitValues.Size();

		if( isBestFeatureDiscrete ) {
			CDecisionTreeDiscreteNodeInfo* info = FINE_DEBUG_NEW CDecisionTreeDiscreteNodeInfo();
			node.SetInfo( info );
			info->FeatureIndex = bestFeature;
			bestSplitValues.MoveTo( info->Values );
			predictions.MoveTo( info->Predictions );
			info->Children.SetBufferSize( bestSplitValues.Size() );
			for( int i = 0; i < info->Values.Size(); i++ ) {
				info->Children.Add( createNode() );
			}
		} else {
			CDecisionTreeContinuousNodeInfo* info = FINE_DEBUG_NEW CDecisionTreeContinuousNodeInfo();
			node.SetInfo( info );
			info->FeatureIndex = bestFeature;
			info->Threshold = bestSplitValues.First();
			info->Child1 = createNode();
			info->Child2 = createNode();
		}

		return true;
	}

	if( logStream != 0 ) {
		*logStream << "Split result: created const node.\n";
	}

	// Now the only option is to create a constant node
	CDecisionTreeConstNodeInfo* info  = FINE_DEBUG_NEW CDecisionTreeConstNodeInfo();
	predictions.MoveTo( info->Predictions );
	node.SetInfo( info );
	return false;
}

// Generates an array of features used
void CDecisionTreeTrainingModel::generateUsedFeatures( int randomSelectedFeaturesCount, int featuresCount,
	CArray<int>& features ) const
{
	features.Empty();
	features.SetBufferSize( featuresCount );
	for( int i = 0; i < featuresCount; i++ ) {
		features.Add( i );
	}

	if( randomSelectedFeaturesCount != NotFound ) {
		NeoAssert( 0 < randomSelectedFeaturesCount );
		NeoAssert( randomSelectedFeaturesCount < featuresCount );
		for( int i = 0; i < randomSelectedFeaturesCount; i++ ) {
			// Pick a random number from [i, featuresCount - 1] range
			int randomInt = ( random == 0 ) ? rand() : random->Next();
			int index = abs( randomInt ) % ( featuresCount - i );
			swap( features[i], features[i + index] );
		}
		features.SetSize( randomSelectedFeaturesCount );
	}
}

// Creates a node
CPtr<CDecisionTreeNodeBase> CDecisionTreeTrainingModel::createNode() const
{
	return FINE_DEBUG_NEW CDecisionTreeClassificationModel();
}

// Creates a node statistics object
CDecisionTreeNodeStatisticBase* CDecisionTreeTrainingModel::createStatistic( CDecisionTreeNodeBase* node ) const
{
	CArray<int> features;
	generateUsedFeatures( params.RandomSelectedFeaturesCount, classificationProblem->GetFeatureCount(), features );
	return FINE_DEBUG_NEW CClassificationStatistics( node, *classificationProblem, features );
}

} // namespace NeoML
