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

#include <DecisionTreeNodeBase.h>

namespace NeoML {

void CDecisionTreeNodeBase::SetInfo( CDecisionTreeNodeInfoBase* newInfo )
{
	if( info != 0 ) {
		delete info;
	}

	info = newInfo;
}

void CDecisionTreeNodeBase::GetClassifyNode( const CFloatVectorDesc& data, CPtr<CDecisionTreeNodeBase>& node, int& additionalLevel ) const
{
	if( info == 0 ) {
		node = const_cast<CDecisionTreeNodeBase*>( this );
		return;
	}

	switch( info->Type ) {
		case DTNT_Discrete:
		{
			CDecisionTreeDiscreteNodeInfo* discreteInfo = static_cast<CDecisionTreeDiscreteNodeInfo*>( info );
			float featureValue = 0;
			GetValue( data, discreteInfo->FeatureIndex, featureValue );

			for( int i = 0; i < discreteInfo->Values.Size(); i++ ) {
				if( discreteInfo->Values[i] == featureValue ) {
					additionalLevel++;
					return discreteInfo->Children[i]->GetClassifyNode( data, node, additionalLevel );
				}
			}
			node = const_cast<CDecisionTreeNodeBase*>( this );
			return;
		}

		case DTNT_Continuous:
		{
			CDecisionTreeContinuousNodeInfo* continuousInfo = static_cast<CDecisionTreeContinuousNodeInfo*>( info );
			float featureValue = 0;
			GetValue( data, continuousInfo->FeatureIndex, featureValue );

			additionalLevel++;
			if( featureValue <= continuousInfo->Threshold ) {
				NeoAssert( continuousInfo->Child1 != 0 );
				return continuousInfo->Child1->GetClassifyNode( data, node, additionalLevel );
			}
			NeoAssert( continuousInfo->Child2 != 0 );
			return continuousInfo->Child2->GetClassifyNode( data, node, additionalLevel );
		}

		case DTNT_Const:
		case DTNT_Undefined:
			node = const_cast<CDecisionTreeNodeBase*>( this );
			return;

		default:
			NeoAssert( false );
	};
}

void CDecisionTreeNodeBase::GetClassifyNode( const CFloatVector& data, CPtr<CDecisionTreeNodeBase>& node, int& additionalLevel ) const
{
	GetClassifyNode( data.GetDesc(), node, additionalLevel );
}

CDecisionTreeNodeBase::~CDecisionTreeNodeBase()
{
	if( info != 0 ) {
		delete info;
	}
}

} // namespace NeoML
