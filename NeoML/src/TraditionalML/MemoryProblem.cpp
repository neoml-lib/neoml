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

#include <NeoML/TraditionalML/MemoryProblem.h>

namespace NeoML {

// The structure for storing vectors
struct CVectorData {
	CSparseFloatVector Vector;
	double Weight;
	int Class;

	CVectorData() : Weight( 0.0 ), Class( NotFound ) {}
	void Serialize( CArchive& archive );
};

void CVectorData::Serialize( CArchive& archive )
{
	archive.Serialize( Weight );
	archive.SerializeSmallValue( Class );
	Vector.Serialize( archive );
}

//------------------------------------------------------------------------------------------------------------

CMemoryProblem::CMemoryProblem( int _featureCount, int _classCount, int rowsBufferSize, int elementsBufferSize ) :
	matrix( _featureCount, rowsBufferSize, elementsBufferSize ),
	classCount( _classCount ),
	featureCount( _featureCount )
{
	if( rowsBufferSize > 0 ) {
		classes.SetBufferSize( rowsBufferSize );
		weights.SetBufferSize( rowsBufferSize );
	}

	NeoAssert( featureCount > 0 );
	isDiscreteFeature.Add( false, featureCount );
	discretizationValues.Add( DefaultDiscretizationValue, featureCount );
}

CMemoryProblem::CMemoryProblem() :
	classCount( 0 ),
	featureCount( NotFound )
{
}

void CMemoryProblem::Add( const CFloatVectorDesc& vector, double weight, int classNumber )
{
	NeoAssert( featureCount > 0 );
	NeoAssert( classCount > classNumber );

	matrix.AddRow( vector );
	classes.Add( classNumber );
	weights.Add( static_cast<float>( weight ) );
}

void CMemoryProblem::Add( const CSparseFloatVector& vector, double weight, int classNumber )
{
	Add( vector.GetDesc(), weight, classNumber );
}

void CMemoryProblem::SetDiscretizationValue( int index, int value )
{
	NeoAssert( 0 <= index && index < featureCount );
	NeoAssert( value > 1 );

	discretizationValues[index] = value;
}

void CMemoryProblem::SetVectorWeight( int index, float newWeight )
{
	NeoAssert( 0 <= index && index < GetVectorCount() );
	weights[index] = newWeight;
}

void CMemoryProblem::SetClass( int index, int newClass )
{
	NeoAssert( 0 <= index && index < GetVectorCount() );
	NeoAssert( newClass >= 0 );
	NeoAssert( classCount > newClass );
	classes[index] = newClass;
}

void CMemoryProblem::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( 1 );
	archive.Serialize( featureCount );
	archive.Serialize( classCount );
	isDiscreteFeature.Serialize( archive );
	discretizationValues.Serialize( archive );

	if( archive.IsLoading() ) {
		if( version > 0 ) {
			archive >> matrix;
			archive >> weights;
			archive >> classes;
		} else {
			unsigned int count;
			archive >> count;
			matrix = CSparseFloatMatrix( featureCount, count );
			CVectorData data;
			for( int i = 0; i < static_cast<int>(count); i++ ) {
				data.Serialize( archive );
				Add( data.Vector, data.Weight, data.Class );
			}
		}
	} else if( archive.IsStoring() ) {
		archive << matrix;
		archive << weights;
		archive << classes;
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
