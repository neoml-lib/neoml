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

#include <OneVersusAllModel.h>

namespace NeoML {

IOneVersusAllModel::~IOneVersusAllModel()
{}

REGISTER_NEOML_MODEL( COneVersusAllModel, OneVersusAllModelName )

COneVersusAllModel::COneVersusAllModel( CObjectArray<IModel>& _classifiers )
{
	NeoAssert( !_classifiers.IsEmpty() );
	_classifiers.MoveTo( classifiers );
}

int COneVersusAllModel::GetClassCount() const
{
	return classifiers.Size();
}

bool COneVersusAllModel::ClassifyEx( const CSparseFloatVector& data,
	COneVersusAllClassificationResult& result ) const
{
	return ClassifyEx( data.GetDesc(), result );
}

bool COneVersusAllModel::ClassifyEx( const CSparseFloatVectorDesc& data,
	COneVersusAllClassificationResult& result ) const
{
	CArray<double> probability;
	result.SigmoidSum = 0.0;
	int preferedClass = 0;

	for( int i = 0; i < classifiers.Size(); i++ ) {
		CClassificationResult curResult;
		NeoAssert( classifiers[i]->Classify( data, curResult ) );
		const double curClassProbability = curResult.Probabilities[0].GetValue();
		probability.Add( curClassProbability );
		result.SigmoidSum += curClassProbability;

		if( curClassProbability > probability[preferedClass] ) {
			preferedClass = i;
		}
	}

	result.ExceptionProbability = CClassificationProbability( 0 );
	result.PreferredClass = preferedClass;
	result.Probabilities.SetSize( probability.Size() );
	for( int i = 0; i < probability.Size(); i++ ) {
		result.Probabilities[i] = CClassificationProbability( probability[i] / result.SigmoidSum );
	}
	return true;
}

bool COneVersusAllModel::ClassifyEx( const CFloatVector& data, COneVersusAllClassificationResult& result ) const
{
	CArray<double> probability;
	result.SigmoidSum = 0.0;
	int preferedClass = 0;

	for( int i = 0; i < classifiers.Size(); i++ ) {
		CClassificationResult curResult;
		NeoAssert( classifiers[i]->Classify( data, curResult ) );
		const double curClassProbability = curResult.Probabilities[0].GetValue();
		probability.Add( curClassProbability );
		result.SigmoidSum += curClassProbability;

		if( curClassProbability > probability[preferedClass] ) {
			preferedClass = i;
		}
	}

	result.ExceptionProbability = CClassificationProbability( 0 );
	result.PreferredClass = preferedClass;
	result.Probabilities.SetSize( probability.Size() );
	for( int i = 0; i < probability.Size(); i++ ) {
		result.Probabilities[i] = CClassificationProbability( probability[i] / result.SigmoidSum );
	}
	return true;
}

bool COneVersusAllModel::Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const
{
	COneVersusAllClassificationResult extendedResult;
	ClassifyEx( data, extendedResult );
	result.ExceptionProbability = extendedResult.ExceptionProbability;
	result.PreferredClass = extendedResult.PreferredClass;
	extendedResult.Probabilities.MoveTo( result.Probabilities );
	return true;
}

bool COneVersusAllModel::Classify( const CFloatVector& data, CClassificationResult& result ) const
{
	COneVersusAllClassificationResult extendedResult;
	ClassifyEx( data, extendedResult );
	result.ExceptionProbability = extendedResult.ExceptionProbability;
	result.PreferredClass = extendedResult.PreferredClass;
	extendedResult.Probabilities.MoveTo( result.Probabilities );
	return true;
}

void COneVersusAllModel::Serialize( CArchive& archive )
{
#ifdef NEOML_USE_FINEOBJ
	const int minSupportedVersion = 0;
#else
	const int minSupportedVersion = 1;
#endif

	int version = archive.SerializeVersion( 1, minSupportedVersion );

	if( archive.IsStoring() ) {
		archive << classifiers.Size();
		for( int i = 0; i < classifiers.Size(); i++ ) {
			archive << CString( GetModelName( classifiers[i] ) );
			classifiers[i]->Serialize( archive );
		}
	} else if( archive.IsLoading() ) {
		int size = 0;
		archive >> size;
		classifiers.SetSize( size );
		for( int i = 0; i < classifiers.Size(); ++i ) {
#ifdef NEOML_USE_FINEOBJ
			if( version == 0 ) {
				CUnicodeString name = archive.ReadExternalName();
				classifiers[i] = CreateModel<IModel>( name.CreateString() );
			}
#endif
			if( version == 1 ) {
				CString name;
				archive >> name;
				classifiers[i] = CreateModel<IModel>( name );
			}
			classifiers[i]->Serialize( archive );
		}
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
