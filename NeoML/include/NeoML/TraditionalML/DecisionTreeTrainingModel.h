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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/ClassificationResult.h>
#include <NeoML/TraditionalML/TrainingModel.h>
#include <NeoML/Random.h>

namespace NeoML {

class CDecisionTreeNodeBase;
class CDecisionTreeNodeStatisticBase;

// The node types for a decision tree
enum TDecisionTreeNodeType {
	DTNT_Undefined = 0,
	DTNT_Const = 1, // a constant node
	DTNT_Discrete = 3, // a node that uses a discrete feature for splitting
	DTNT_Continuous = 4 // a node that uses a continuous feature for splitting
};

// The information about a decision tree node
struct CDecisionTreeNodeInfo {
	TDecisionTreeNodeType Type; // the type of the node
	// The index of the feature which is used for splitting
	// This value makes sense only for nodes with DTNT_Discrete and DTNT_Continuous types
	int FeatureIndex;
	// The values which are used for node splitting (valid only for DTNT_Discrete and DTNT_Continuous)
	// For DTNT_Discrete, if the feature value is found in the Values array the child node with the corresponding index is selected;
	// otherwise, the class is determined using the Probabilities field
	CArray<double> Values;
	// The probabilities for the object to belong to each of the classes
	// This field makes sense only for DTNT_Discrete and DTNT_Const
	// For DTNT_Const, this field is used to classify the input
	// For DTNT_Discrete, this field is used to classify the input if the feature value has not been found in Values
	CArray<CClassificationProbability> Probabilities;

	CDecisionTreeNodeInfo() : Type( DTNT_Undefined ), FeatureIndex( NotFound ) {}

	// Copies the contents to another object
	void CopyTo( CDecisionTreeNodeInfo& newInfo ) const;
};

inline void CDecisionTreeNodeInfo::CopyTo( CDecisionTreeNodeInfo& newInfo ) const
{
	newInfo.Type = Type;
	newInfo.FeatureIndex = FeatureIndex;
	Values.CopyTo( newInfo.Values );
	Probabilities.CopyTo( newInfo.Probabilities );
}

inline CArchive& operator<<( CArchive& archive, const CDecisionTreeNodeInfo& info )
{
	archive.SerializeEnum( const_cast<CDecisionTreeNodeInfo&>( info ).Type );
	archive << info.FeatureIndex;
	archive << info.Values;
	archive << info.Probabilities;
	return archive;
}

inline CArchive& operator>>( CArchive& archive, CDecisionTreeNodeInfo& info )
{
	archive.SerializeEnum( info.Type );
	archive >> info.FeatureIndex;
	archive >> info.Values;
	archive >> info.Probabilities;
	return archive;
}

//------------------------------------------------------------------------------------------------------------

DECLARE_NEOML_MODEL_NAME( DecisionTreeModelName, "FmlDecisionTreeClassificationModel" )

// Classification model interface
class NEOML_API IDecisionTreeModel : public IModel {
public:
	virtual ~IDecisionTreeModel();

	// Gets the number of child nodes
	virtual int GetChildrenCount() const = 0;

	// Gets the child node with the specified index
	virtual CPtr<IDecisionTreeModel> GetChild( int index ) const = 0;

	// Gets the node information
	virtual void GetNodeInfo( CDecisionTreeNodeInfo& info ) const = 0;
};

//------------------------------------------------------------------------------------------------------------

// Decision tree training algorithm
class NEOML_API CDecisionTreeTrainingModel : public ITrainingModel {
public:
	// The type of criterion to be used for subtree splitting
	enum TSplitCriterion {
		SC_GiniImpurity = 0,
		SC_InformationGain,
		SC_Count
	};

	// Classification parameters
	struct CParams {
		// The minimum number of vectors corresponding to a node subtree:
		int MinContinuousSubsetSize; // when splitting by a continuous feature value
		int MinDiscreteSubsetSize; // when splitting by a discrete feature value
		// The minimum weight of the vectors in a subtree relative to the parent node weight (may be from 0 to 1):
		double MinDiscreteSubsetPart; // when splitting by a discrete feature value
		double MinContinuousSubsetPart; // when splitting by a continuous feature value
		// The minimum number of vectors in a node subtree when it may be divided further
		int MinSplitSize;
		int MaxTreeDepth; // the maximum depth of the tree
		// The maximum number of nodes in the tree
		int MaxNodesCount;
		// The type of split criterion to be used for splitting nodes
		TSplitCriterion SplitCriterion;
		// If the ratio of same class elements in the subset is greater than this value, a constant node will be created
		// May be from 0 to 1
		double ConstNodeThreshold;
		// No more than this number of randomly selected features will be used for each node
		// Set the value to `-1` to use all features every time
		int RandomSelectedFeaturesCount;
		// The memory limit for the algorithm
		size_t AvailableMemory; 

		CParams() :
			MinContinuousSubsetSize( 1 ),
			MinDiscreteSubsetSize( 1 ),
			MinDiscreteSubsetPart( 0 ),
			MinContinuousSubsetPart( 0 ),
			MinSplitSize( 1 ),
			MaxTreeDepth( 32 ),
			MaxNodesCount( 4096 ),
			SplitCriterion( SC_InformationGain ),
			ConstNodeThreshold( 0.99 ),
			RandomSelectedFeaturesCount( NotFound ),
			AvailableMemory( Gigabyte )
		{
		}
	};

	// All features will be used
	explicit CDecisionTreeTrainingModel( const CParams& params, CRandom* random = 0 );
	~CDecisionTreeTrainingModel();

	// Set a text stream to log the progress
	void SetLog( CTextStream* newLog ) { logStream = newLog; }

	// The ITrainingModel interface methods:
	virtual CPtr<IModel> Train( const IProblem& problem );

private:
	static const int MaxClassifyNodesCacheSize = 10 * Megabyte; // the cache size for leaf nodes
	CParams params; // the classification parameters
	CRandom* random; // the random numbers generator
	CTextStream* logStream; // the logging stream
	CPtr<const IProblem> classificationProblem; // the current input data as an IProblem interface
	mutable int nodesCount; // the number of tree nodes
	mutable int statisticsCacheSize; // the cache size for statistics
	mutable CPointerArray<CDecisionTreeNodeStatisticBase> statisticsCache; // the cache for statistics
	mutable CArray<CDecisionTreeNodeBase*> classifyNodesCache; // the cache for leaf nodes
	mutable CArray<int> classifyNodesLevel; // the levels of leaf nodes

	CPtr<CDecisionTreeNodeBase> buildTree( int vectorCount );
	bool buildTreeLevel( const CFloatMatrixDesc& matrix, int level, CDecisionTreeNodeBase& root ) const;
	bool collectStatistics( const CFloatMatrixDesc& matrix, int level, CDecisionTreeNodeBase* root ) const;
	bool split( const CDecisionTreeNodeStatisticBase& nodeStatistics, int level ) const;
	void generateUsedFeatures( int randomSelectedFeaturesCount, int featuresCount, CArray<int>& features ) const;

	CPtr<CDecisionTreeNodeBase> createNode() const;
	CDecisionTreeNodeStatisticBase* createStatistic( CDecisionTreeNodeBase* node ) const;
};

} // namespace NeoML
