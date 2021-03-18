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

#include <DecisionTreeClassificationModel.h>

namespace NeoML {

REGISTER_NEOML_MODEL( CDecisionTreeClassificationModel, DecisionTreeModelName )

// Defines the most probable class
static int definePreferredClass( const CArray<double>& probabilities )
{
	int result = 0;
	for( int i = 1; i < probabilities.Size(); i++ ) {
		if( probabilities[i] > probabilities[result] ) {
			result = i;
		}
	}
	return result;
}

//------------------------------------------------------------------------------------------------------------

int CDecisionTreeClassificationModel::GetChildrenCount() const
{
	CDecisionTreeNodeInfoBase* info = GetInfo();
	NeoAssert( info != 0 );

	if( info->Type == DTNT_Discrete ) {
		CDecisionTreeDiscreteNodeInfo* discreteInfo = static_cast<CDecisionTreeDiscreteNodeInfo*>( info );
		return discreteInfo->Children.Size();
	}
	if( info->Type == DTNT_Continuous ) {
		return 2;
	}
	return 0;
}

CPtr<IDecisionTreeModel> CDecisionTreeClassificationModel::GetChild( int index ) const
{
	CDecisionTreeNodeInfoBase* info = GetInfo();
	NeoAssert( info != 0 );

	if( info->Type == DTNT_Discrete ) {
		CDecisionTreeDiscreteNodeInfo* discreteInfo = static_cast<CDecisionTreeDiscreteNodeInfo*>( info );
		return dynamic_cast<IDecisionTreeModel*>( discreteInfo->Children[index].Ptr() );
	}
	if( info->Type == DTNT_Continuous ) {
		CDecisionTreeContinuousNodeInfo* continuousInfo = static_cast<CDecisionTreeContinuousNodeInfo*>( info );
		if( index == 0 ) {
			return dynamic_cast<IDecisionTreeModel*>( continuousInfo->Child1.Ptr() );
		} else if( index == 1 ) {
			return dynamic_cast<IDecisionTreeModel*>( continuousInfo->Child2.Ptr() );
		}
		NeoAssert( false );
	}
	return 0;
}

void CDecisionTreeClassificationModel::GetNodeInfo( CDecisionTreeNodeInfo& result ) const
{
	CDecisionTreeNodeInfoBase* info = GetInfo();
	NeoAssert( info != 0 );

	switch( info->Type ) {
		case DTNT_Const:
		{
			CDecisionTreeConstNodeInfo* constInfo = static_cast<CDecisionTreeConstNodeInfo*>( info );
			result.FeatureIndex = NotFound;
			result.Probabilities.Empty();
			for( int i = 0; i < constInfo->Predictions.Size(); i++ ) {
				result.Probabilities.Add( CClassificationProbability( constInfo->Predictions[i] ) );
			}
			result.Values.Empty();
			result.Type = DTNT_Const;
			break;
		}
		case DTNT_Discrete:
		{
			CDecisionTreeDiscreteNodeInfo* discreteInfo = static_cast<CDecisionTreeDiscreteNodeInfo*>( info );
			result.FeatureIndex = discreteInfo->FeatureIndex;
			result.Probabilities.Empty();
			for( int i = 0; i < discreteInfo->Predictions.Size(); i++ ) {
				result.Probabilities.Add( CClassificationProbability( discreteInfo->Predictions[i] ) );
			}
			discreteInfo->Values.CopyTo( result.Values );
			result.Type = DTNT_Discrete;
			break;
		}
		case DTNT_Continuous:
		{
			CDecisionTreeContinuousNodeInfo* continuousInfo = static_cast<CDecisionTreeContinuousNodeInfo*>( info );
			result.FeatureIndex = continuousInfo->FeatureIndex;
			result.Probabilities.Empty();
			result.Values.SetSize( 2 );
			result.Values[0] = continuousInfo->Threshold;
			result.Values[1] = DBL_MAX;
			result.Type = DTNT_Continuous;
			break;
		}

		default:
			NeoAssert( false );
			break;
	};
}

int CDecisionTreeClassificationModel::GetClassCount() const
{
	CDecisionTreeNodeInfoBase* info = GetInfo();
	NeoAssert( info != 0 );

	switch( info->Type ) {
		case DTNT_Const:
		{
			CDecisionTreeConstNodeInfo* constInfo = static_cast<CDecisionTreeConstNodeInfo*>( info );
			return constInfo->Predictions.Size();
		}
		case DTNT_Discrete:
		{
			CDecisionTreeDiscreteNodeInfo* discreteInfo = static_cast<CDecisionTreeDiscreteNodeInfo*>( info );
			return discreteInfo->Predictions.Size();
		}
		case DTNT_Continuous:
		{
			CDecisionTreeContinuousNodeInfo* continuousInfo = static_cast<CDecisionTreeContinuousNodeInfo*>( info );
			NeoAssert( continuousInfo->Child1 != 0 );
			return dynamic_cast<CDecisionTreeClassificationModel*>( continuousInfo->Child1.Ptr() )->GetClassCount();
		}

		default:
			NeoAssert( false );
			return 0;
	};
}

bool CDecisionTreeClassificationModel::Classify( const CFloatVectorDesc& data, CClassificationResult& result ) const
{
	CPtr<CDecisionTreeNodeBase> node;
	int level = 0;
	GetClassifyNode( data, node, level );

	return classify( node, result );
}

bool CDecisionTreeClassificationModel::Classify( const CFloatVector& data, CClassificationResult& result ) const
{
	CPtr<CDecisionTreeNodeBase> node;
	int level = 0;
	GetClassifyNode( data, node, level );

	return classify( node, result );
}

void CDecisionTreeClassificationModel::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );

	CDecisionTreeNodeInfoBase* info = GetInfo();
	TDecisionTreeNodeType type = ( info == 0 ? DTNT_Undefined : info->Type );
	archive.SerializeEnum( type );
	
	if( archive.IsStoring() ) {
		switch( type ) {
			case DTNT_Const:
			{
				CDecisionTreeConstNodeInfo* constInfo = static_cast<CDecisionTreeConstNodeInfo*>( info );
				constInfo->Serialize( archive );
				break;
			}
			case DTNT_Discrete:
			{
				CDecisionTreeDiscreteNodeInfo* discreteInfo = static_cast<CDecisionTreeDiscreteNodeInfo*>( info );
				discreteInfo->Serialize( archive );
				break;
			}
			case DTNT_Continuous:
			{
				CDecisionTreeContinuousNodeInfo* continuousInfo = static_cast<CDecisionTreeContinuousNodeInfo*>( info );
				continuousInfo->Serialize( archive );
				break;
			}
			case DTNT_Undefined:
				break;

			default:
				NeoAssert( false );
				break;
		};
	} else if( archive.IsLoading() ) {
		switch( type ) {
			case DTNT_Const:
			{
				CDecisionTreeConstNodeInfo* constInfo = FINE_DEBUG_NEW CDecisionTreeConstNodeInfo();
				SetInfo( constInfo );
				constInfo->Serialize( archive );
				break;
			}
			case DTNT_Discrete:
			{
				CDecisionTreeDiscreteNodeInfo* discreteInfo = FINE_DEBUG_NEW CDecisionTreeDiscreteNodeInfo();
				SetInfo( discreteInfo );
				discreteInfo->Serialize( archive );
				break;
			}
			case DTNT_Continuous:
			{
				CDecisionTreeContinuousNodeInfo* continuousInfo = FINE_DEBUG_NEW CDecisionTreeContinuousNodeInfo();
				SetInfo( continuousInfo );
				continuousInfo->Serialize( archive );
				break;
			}
			case DTNT_Undefined:
				SetInfo( 0 );
				break;

			default:
				NeoAssert( false );
				break;
		};
	} else {
		NeoAssert( false );
	}
}

// Performs classification in a node
bool CDecisionTreeClassificationModel::classify( CDecisionTreeNodeBase* node, CClassificationResult& result ) const
{
	NeoAssert( node != 0 );
	NeoAssert( node->GetInfo() != 0 );

	switch( node->GetInfo()->Type ) {
		case DTNT_Discrete:
		{
			CDecisionTreeDiscreteNodeInfo* discreteInfo = static_cast<CDecisionTreeDiscreteNodeInfo*>( node->GetInfo() );
			result.PreferredClass = definePreferredClass( discreteInfo->Predictions );
			result.ExceptionProbability = CClassificationProbability( 0 );
			result.Probabilities.Empty();
			for( int i = 0; i < discreteInfo->Predictions.Size(); i++ ) {
				result.Probabilities.Add( CClassificationProbability( discreteInfo->Predictions[i] ) );
			}
			return true;
		}
		case DTNT_Const:
		{
			CDecisionTreeConstNodeInfo* constInfo = static_cast<CDecisionTreeConstNodeInfo*>( node->GetInfo() );
			result.PreferredClass = definePreferredClass( constInfo->Predictions );
			result.ExceptionProbability = CClassificationProbability( 0 );
			result.Probabilities.Empty();
			for( int i = 0; i < constInfo->Predictions.Size(); i++ ) {
				result.Probabilities.Add( CClassificationProbability( constInfo->Predictions[i] ) );
			}
			return true;
		}
		default:
			NeoAssert( false );
	};

	return false;
}

} // namespace NeoML
