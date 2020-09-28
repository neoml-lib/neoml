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

#include <NeoML/TraditionalML/DecisionTreeTrainingModel.h>

namespace NeoML {

// Base structure for decision tree node info
struct CDecisionTreeNodeInfoBase {
	TDecisionTreeNodeType Type; // node type

	CDecisionTreeNodeInfoBase( TDecisionTreeNodeType type ) : Type( type ) {}
	virtual ~CDecisionTreeNodeInfoBase() {}
};

// Base class for a decision tree node
class CDecisionTreeNodeBase : public virtual IObject {
public:
	CDecisionTreeNodeBase() : info( 0 ) {}

	TDecisionTreeNodeType GetType() const { return info == 0 ? DTNT_Undefined : info->Type; }

	// Sets the node info
	void SetInfo( CDecisionTreeNodeInfoBase* newInfo );
	CDecisionTreeNodeInfoBase* GetInfo() const { return info; }

	// Gets the node to be used for classification
	void GetClassifyNode( const CSparseFloatVectorDesc& data, CPtr<CDecisionTreeNodeBase>& node, int& level ) const;
	void GetClassifyNode( const CFloatVector& data, CPtr<CDecisionTreeNodeBase>& node, int& level ) const;

protected:
	virtual ~CDecisionTreeNodeBase(); // delete operator prohibited

private:
	CDecisionTreeNodeInfoBase* info; // node info
};

//------------------------------------------------------------------------------------------------------------

// Information about a node of the DTNT_Discrete type
struct CDecisionTreeDiscreteNodeInfo : public CDecisionTreeNodeInfoBase {
	int FeatureIndex; // the index of the feature by which to split
	CArray<double> Values; // the values by which to split
	CArray<double> Predictions; // the class predictions
	CObjectArray<CDecisionTreeNodeBase> Children; // the child nodes corresponding to the values

	CDecisionTreeDiscreteNodeInfo() : CDecisionTreeNodeInfoBase( DTNT_Discrete ) {}

	void Serialize( CArchive& archive );
};

inline void CDecisionTreeDiscreteNodeInfo::Serialize( CArchive& archive )
{
#ifdef NEOML_USE_FINEOBJ
	const int minSupportedVersion = 0;
#else
	const int minSupportedVersion = 1;
#endif

	int version = archive.SerializeVersion( 1, minSupportedVersion );

	if( archive.IsStoring() ) {
		archive << FeatureIndex;
		archive << Values;
		archive << Predictions;
		archive << Children.Size();
		for( int i = 0; i < Children.Size(); i++ ) {
			archive << CString( GetModelName( Children[i] ) );
			Children[i]->Serialize( archive );
		}
	} else if( archive.IsLoading() ) {
		archive >> FeatureIndex;
		archive >> Values;
		archive >> Predictions;

		int size = 0;
		archive >> size;
		Children.SetSize( size );
		for( int i = 0; i < Children.Size(); i++ ) {
#ifdef NEOML_USE_FINEOBJ
			if( version == 0 ) {
				CUnicodeString modelName = archive.ReadExternalName();
				Children[i] = CreateModel<CDecisionTreeNodeBase>( modelName.CreateString() );
			}
#endif
			if( version == 1 ) {
				CString modelName;
				archive >> modelName;
				Children[i] = CreateModel<CDecisionTreeNodeBase>( modelName );
			}

			Children[i]->Serialize( archive );
		}
	} else {
		NeoAssert( false );
	}
}

// Information about a node of the DTNT_Const type
struct CDecisionTreeConstNodeInfo : public CDecisionTreeNodeInfoBase {
	CArray<double> Predictions; // the class predictions

	CDecisionTreeConstNodeInfo() : CDecisionTreeNodeInfoBase( DTNT_Const ) {}

	void Serialize( CArchive& archive );
};

inline void CDecisionTreeConstNodeInfo::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );

	if( archive.IsLoading() ) {
		archive >> Predictions;
	} else {
		archive << Predictions;
	}
}

// Information about a node of the DTNT_Continuous type
struct CDecisionTreeContinuousNodeInfo : public CDecisionTreeNodeInfoBase {
	int FeatureIndex; // the index of the feature by which to split
	double Threshold; // the threshold by which to split
	CPtr<CDecisionTreeNodeBase> Child1;
	CPtr<CDecisionTreeNodeBase> Child2;

	CDecisionTreeContinuousNodeInfo() : CDecisionTreeNodeInfoBase( DTNT_Continuous ) {}

	void Serialize( CArchive& archive );
};

inline void CDecisionTreeContinuousNodeInfo::Serialize( CArchive& archive )
{
#ifdef NEOML_USE_FINEOBJ
	const int minSupportedVersion = 0;
#else
	const int minSupportedVersion = 1;
#endif

	int version = archive.SerializeVersion( 1, minSupportedVersion );

	if( archive.IsStoring() ) {
		archive << FeatureIndex;
		archive << Threshold;
		archive << CString( GetModelName( Child1 ) );
		Child1->Serialize( archive );
		archive << CString( GetModelName( Child2 ) );
		Child2->Serialize( archive );
	} else if( archive.IsLoading() ) {
		archive >> FeatureIndex;
		archive >> Threshold;
#ifdef NEOML_USE_FINEOBJ
		if( version == 0 ) {
			CUnicodeString name = archive.ReadExternalName();
			Child1 = CreateModel<CDecisionTreeNodeBase>( name.CreateString() );
			Child1->Serialize( archive );
			name = archive.ReadExternalName();
			Child2 = CreateModel<CDecisionTreeNodeBase>( name.CreateString() );
			Child2->Serialize( archive );
		}
#endif
		if( version == 1 ) {
			CString name;
			archive >> name;
			Child1 = CreateModel<CDecisionTreeNodeBase>( name );
			Child1->Serialize( archive );
			archive >> name;
			Child2 = CreateModel<CDecisionTreeNodeBase>( name );
			Child2->Serialize( archive );
		}
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
