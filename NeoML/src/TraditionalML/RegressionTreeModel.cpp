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

#include <RegressionTreeModel.h>

namespace NeoML {

REGISTER_NEOML_MODEL( CRegressionTreeModel, RegressionTreeModelName )

// Serializes an integer depending on its value (similar to UTF encoding)
static void serializeCompact( CArchive& archive, unsigned int& value )
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

CRegressionTreeModel::CRegressionTreeModel()
{
	info.Type = RTNT_Undefined;
	info.FeatureIndex = NotFound;
	info.Value = 0;
}

CRegressionTreeModel::~CRegressionTreeModel()
{
	NeoPresume( info.Type != RTNT_Undefined );
}

void CRegressionTreeModel::InitLeafNode( double prediction )
{
	info.Type = RTNT_Const;
	info.FeatureIndex = NotFound;
	info.Value = prediction;
	leftChild.Release();
	rightChild.Release();
}

void CRegressionTreeModel::InitSplitNode( CRegressionTreeModel& left, CRegressionTreeModel& right, int feature, double threshold )
{
	NeoAssert( info.Type == RTNT_Undefined );

	info.Type = RTNT_Continuous;
	info.FeatureIndex = feature;
	info.Value = threshold;
	leftChild = &left;
	rightChild = &right;
}

const CRegressionTreeModel* CRegressionTreeModel::GetPredictionNode( const CSparseFloatVector& data ) const
{
	static_assert( RTNT_Count == 3, "RTNT_Count != 3" );

	if( info.Type == RTNT_Continuous ) {
		float featureValue = 0;
		data.GetValue( info.FeatureIndex, featureValue );

		const CRegressionTreeModel* child = featureValue <= info.Value ? leftChild : rightChild;
		NeoAssert( child != 0 );
		return child->GetPredictionNode( data );
	}
	return this;
}

const CRegressionTreeModel* CRegressionTreeModel::GetPredictionNode( const CFloatVector& data ) const
{
	static_assert( RTNT_Count == 3, "RTNT_Count != 3" );

	if( info.Type == RTNT_Continuous ) {
		double featureValue = info.FeatureIndex < data.Size() ? data[info.FeatureIndex] : 0;
		const CRegressionTreeModel* child = featureValue <= info.Value ? leftChild : rightChild;
		NeoAssert( child != 0 );
		return child->GetPredictionNode( data );
	}
	return this;
}

const CRegressionTreeModel* CRegressionTreeModel::GetPredictionNode( const CSparseFloatVectorDesc& data ) const
{
	static_assert( RTNT_Count == 3, "RTNT_Count != 3" );

	if( info.Type == RTNT_Continuous ) {
		float featureValue = 0;
		GetValue( data, info.FeatureIndex, featureValue );

		const CRegressionTreeModel* child = featureValue <= info.Value ? leftChild : rightChild;
		NeoAssert( child != 0 );
		return child->GetPredictionNode( data );
	}
	return this;
}

void CRegressionTreeModel::CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const
{
	result.Empty();
	result.Add( 0, maxFeature );

	calcFeatureStatistics( maxFeature, result );
}

double CRegressionTreeModel::Predict( const CSparseFloatVector& data ) const
{
	const CRegressionTreeModel* node = GetPredictionNode( data );
	NeoAssert( node->info.Type == RTNT_Const );
	return node->info.Value;
}

double CRegressionTreeModel::Predict( const CFloatVector& data ) const
{
	const CRegressionTreeModel* node = GetPredictionNode( data );
	NeoAssert( node->info.Type == RTNT_Const );
	return node->info.Value;
}

double CRegressionTreeModel::Predict( const CSparseFloatVectorDesc& data ) const
{
	const CRegressionTreeModel* node = GetPredictionNode( data );
	NeoAssert( node->info.Type == RTNT_Const );
	return node->info.Value;
}

void CRegressionTreeModel::Serialize( CArchive& archive )
{
#ifdef NEOML_USE_FINEOBJ
	const int minSupportedVersion = 0;
#else
	const int minSupportedVersion = 1;
#endif
	int version = archive.SerializeVersion( 2, minSupportedVersion );

	if( archive.IsStoring() ) {
		unsigned int index = info.FeatureIndex == NotFound ? 0 : info.FeatureIndex + 1;
		serializeCompact( archive, index );
		archive << info.Value;
		if( info.Type == RTNT_Continuous ) {
			NeoAssert( leftChild != 0 );
			leftChild->Serialize( archive );
			NeoAssert( rightChild != 0 );
			rightChild->Serialize( archive );
		}
	} else if( archive.IsLoading() ) {
		switch( version ) {
#ifdef NEOML_USE_FINEOBJ
			case 0:
			{
				archive >> info;
				if( info.Type == RTNT_Continuous ) {
					CUnicodeString name = archive.ReadExternalName();
					leftChild = FINE_DEBUG_NEW CRegressionTreeModel();
					leftChild->Serialize( archive );
					name = archive.ReadExternalName();
					rightChild = FINE_DEBUG_NEW CRegressionTreeModel();
					rightChild->Serialize( archive );
				}
				break;
			}
#endif
			case 1:
			case 2:
			{
				unsigned int index = 0;
				serializeCompact( archive, index );
				if( version == 1 ) {
					float value = 0;
					archive >> value;
					info.Value = value;
				} else {
					archive >> info.Value;
				}
				if( index > 0 ) {
					info.Type = RTNT_Continuous;
					info.FeatureIndex = index - 1;
					leftChild = FINE_DEBUG_NEW CRegressionTreeModel();
					leftChild->Serialize( archive );
					rightChild = FINE_DEBUG_NEW CRegressionTreeModel();
					rightChild->Serialize( archive );
				} else {
					info.Type = RTNT_Const;
					info.FeatureIndex = NotFound;
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
void CRegressionTreeModel::calcFeatureStatistics( int maxFeature, CArray<int>& result ) const
{
	static_assert( RTNT_Count == 3, "RTNT_Count != 3" );

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
			return;

		default:
			NeoAssert( false );
	};
}

} // namespace NeoML
