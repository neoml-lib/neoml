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

#include <LinkedRegressionTree.h>
#include <SerializeCompact.h>

namespace NeoML {

REGISTER_NEOML_MODEL( CLinkedRegressionTree, "FmlRegressionTreeModel" )

//------------------------------------------------------------------------------------------------------------

CLinkedRegressionTree::CLinkedRegressionTree()
{
	info.Type = RTNT_Undefined;
	info.FeatureIndex = NotFound;
	info.Value = { 0 };
}

CLinkedRegressionTree::~CLinkedRegressionTree()
{
	NeoPresume( info.Type != RTNT_Undefined );
}

void CLinkedRegressionTree::InitLeafNode( double prediction )
{
	info.Type = RTNT_Const;
	info.FeatureIndex = NotFound;
	info.Value = { prediction };
	leftChild.Release();
	rightChild.Release();
}

void CLinkedRegressionTree::InitLeafNode( const CArray<double>& prediction )
{
	info.Type = RTNT_MultiConst;
	info.FeatureIndex = NotFound;
	info.Value.SetSize( prediction.Size() );
	for( int i = 0; i < prediction.Size(); i++ ) {
		info.Value[i] = prediction[i];
	}
	leftChild.Release();
	rightChild.Release();
}

void CLinkedRegressionTree::InitSplitNode(
	CLinkedRegressionTree& left, CLinkedRegressionTree& right, int feature, double threshold )
{
	NeoAssert( info.Type == RTNT_Undefined );

	info.Type = RTNT_Continuous;
	info.FeatureIndex = feature;
	info.Value = { threshold };
	leftChild = &left;
	rightChild = &right;
}

const CLinkedRegressionTree* CLinkedRegressionTree::GetPredictionNode( const CSparseFloatVectorDesc& data ) const
{
	static_assert(RTNT_Count == 4, "RTNT_Count != 4");

	if( info.Type == RTNT_Continuous ) {
		float featureValue = 0;
		GetValue( data, info.FeatureIndex, featureValue );

		const CLinkedRegressionTree* child = featureValue <= info.Value[0] ? leftChild : rightChild;
		NeoAssert( child != 0 );
		return child->GetPredictionNode( data );
	}
	return this;
}

const CLinkedRegressionTree* CLinkedRegressionTree::GetPredictionNode( const CFloatVector& data ) const
{
	static_assert( RTNT_Count == 4, "RTNT_Count != 4" );

	if( info.Type == RTNT_Continuous ) {
		double featureValue = info.FeatureIndex < data.Size() ? data[info.FeatureIndex] : 0;
		const CLinkedRegressionTree* child = featureValue <= info.Value[0] ? leftChild : rightChild;
		NeoAssert( child != 0 );
		return child->GetPredictionNode( data );
	}
	return this;
}

void CLinkedRegressionTree::CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const
{
	result.Empty();
	result.Add( 0, maxFeature );

	calcFeatureStatistics( maxFeature, result );
}

void CLinkedRegressionTree::Predict( const CFloatVector& data, CPrediction& result ) const
{
	const CLinkedRegressionTree* node = GetPredictionNode( data );
	NeoAssert( node->info.Type == RTNT_MultiConst || node->info.Type == RTNT_Const );
	node->info.Value.CopyTo( result );
}

void CLinkedRegressionTree::Predict( const CSparseFloatVectorDesc& data, CPrediction& result ) const
{
	const CLinkedRegressionTree* node = GetPredictionNode( data );
	NeoAssert( node->info.Type == RTNT_MultiConst || node->info.Type == RTNT_Const );
	node->info.Value.CopyTo( result );
}

void CLinkedRegressionTree::Serialize( CArchive& archive )
{
#ifdef NEOML_USE_FINEOBJ
	const int minSupportedVersion = 0;
#else
	const int minSupportedVersion = 1;
#endif
	int version = archive.SerializeVersion( 3, minSupportedVersion );

	if( archive.IsStoring() ) {
		if( info.Type == RTNT_Continuous ) {
			unsigned int index = static_cast<unsigned int>( info.FeatureIndex + 2 );
			SerializeCompact( archive, index );
			archive << info.Value[0];
			NeoAssert( leftChild != 0 );
			leftChild->Serialize( archive );
			NeoAssert( rightChild != 0 );
			rightChild->Serialize( archive );
		} else if( info.Type == RTNT_Const ) {
			unsigned int type = 0;
			SerializeCompact( archive, type );
			archive << info.Value[0];
		} else if( info.Type == RTNT_MultiConst ) {
			unsigned int type = 1;
			SerializeCompact( archive, type );
			info.Value.Serialize( archive );
		}
	} else if( archive.IsLoading() ) {
		switch( version ) {
#ifdef NEOML_USE_FINEOBJ
			case 0:
			{
				archive >> info;
				if( info.Type == RTNT_Continuous ) {
					CUnicodeString name = archive.ReadExternalName();
					leftChild = FINE_DEBUG_NEW CLinkedRegressionTree();
					leftChild->Serialize( archive );
					name = archive.ReadExternalName();
					rightChild = FINE_DEBUG_NEW CLinkedRegressionTree();
					rightChild->Serialize( archive );
				}
				break;
			}
#endif
			case 1:
			case 2:
			{
				unsigned int index = 0;
				SerializeCompact( archive, index );
				if( version == 1 ) {
					float value = 0;
					archive >> value;
					info.Value = { static_cast<double>( value ) };
				} else {
					double value = 0;
					archive >> value;
					info.Value = { value };
				}
				if( index > 0 ) {
					info.Type = RTNT_Continuous;
					info.FeatureIndex = index - 1;
					leftChild = FINE_DEBUG_NEW CLinkedRegressionTree();
					leftChild->Serialize( archive );
					rightChild = FINE_DEBUG_NEW CLinkedRegressionTree();
					rightChild->Serialize( archive );
				} else {
					info.Type = RTNT_Const;
					info.FeatureIndex = NotFound;
				}
				break;
			}
			case 3:
			{
				unsigned int index = 0;
				SerializeCompact( archive, index );
				if( index >= 2 ) {
					info.Type = RTNT_Continuous;
					info.FeatureIndex = index - 2;
					double value;
					archive >> value;
					info.Value = { value };
					leftChild = FINE_DEBUG_NEW CLinkedRegressionTree();
					leftChild->Serialize( archive );
					rightChild = FINE_DEBUG_NEW CLinkedRegressionTree();
					rightChild->Serialize( archive );
				} else if( index == 0 ) {
					info.Type = RTNT_Const;
					info.FeatureIndex = NotFound;
					double value;
					archive >> value;
					info.Value = { value };
				} else if( index == 1 ) {
					info.Type = RTNT_MultiConst;
					info.FeatureIndex = NotFound;
					info.Value.Serialize( archive );
				}
				break;
			}
			default:
				NeoAssert( false );
		}
	} else {
		NeoAssert( false );
	}
}

// Calculates the feature use frequency
void CLinkedRegressionTree::calcFeatureStatistics( int maxFeature, CArray<int>& result ) const
{
	static_assert( RTNT_Count == 4, "RTNT_Count != 4" );

	switch( info.Type ) {
		case RTNT_Continuous:
		{
			if( info.FeatureIndex < maxFeature ) {
				result[info.FeatureIndex]++;
			}
			leftChild->calcFeatureStatistics( maxFeature, result );
			rightChild->calcFeatureStatistics( maxFeature, result );
			return;
		}

		case RTNT_Undefined: // both leaves
		case RTNT_Const:
		case RTNT_MultiConst:
			return;

		default:
			NeoAssert( false );
	};
}

} // namespace NeoML
