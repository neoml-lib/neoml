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

#include <SvmBinaryModel.h>

namespace NeoML {

ISvmBinaryModel::~ISvmBinaryModel()
{
}

REGISTER_NEOML_MODEL( CSvmBinaryModel, SvmBinaryModelName )

CSvmBinaryModel::CSvmBinaryModel( const CSvmKernel& _kernel, const IProblem& problem, const CArray<double>& _alpha,
		double _freeTerm ) :
	kernel( _kernel ),
	freeTerm( _freeTerm )
{
	CFloatVectorDesc desc;
	CFloatMatrixDesc problemMatrix = problem.GetMatrix();
	NeoAssert( problemMatrix.Height == problem.GetVectorCount() );
	NeoAssert( problemMatrix.Width == problem.GetFeatureCount() );

	int totalVectorSize = 0;
	for( int i = 0; i < _alpha.Size(); i++ ) {
		if( _alpha[i] != 0.0 ) {
			alpha.Add( _alpha[i] * problem.GetBinaryClass( i ) );
			problemMatrix.GetRow( i, desc );
			totalVectorSize += desc.Size;
		}
	}

	matrix = CSparseFloatMatrix( problemMatrix.Width, alpha.Size(), totalVectorSize );
	for( int i = 0; i < _alpha.Size(); i++ ) {
		if( _alpha[i] != 0.0 ) {
			problemMatrix.GetRow( i, desc );
			matrix.AddRow( desc );
		}
	}
}

bool CSvmBinaryModel::Classify( const CFloatVectorDesc& data, CClassificationResult& result ) const
{
	CFloatVectorDesc desc;
	double value = freeTerm;
	for( int i = 0; i < alpha.Size(); i++ ) {
		matrix.GetRow( i, desc );
		value += alpha[i] * kernel.Calculate( data, desc );
	}

	const double probability = 1 / ( 1 + exp( value ) );
	result.ExceptionProbability = CClassificationProbability( 0 );
	result.Probabilities.SetSize( 2 );
	result.Probabilities[0] = CClassificationProbability( probability );
	result.Probabilities[1] = CClassificationProbability( 1 - probability );

	if( probability > 1 - probability ) {
		result.PreferredClass = 0;
	} else {
		result.PreferredClass = 1;
	}
	return true;
}

void CSvmBinaryModel::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( 1 );

	if( archive.IsStoring() ) {
		archive << kernel;
		archive << freeTerm;
		archive << matrix;
		archive << alpha;
	} else if( archive.IsLoading() ) {
		if( version > 0 ) {
			archive >> kernel;
			archive >> freeTerm;
			archive >> matrix;
			archive >> alpha;
		} else {
			NeoAssert( kernel.KernelType() == CSvmKernel::KT_Undefined );
			NeoAssert( freeTerm == 0 );
			archive >> kernel;
			archive >> freeTerm;
			unsigned int count = 0;
			archive >> count;
			CSparseFloatVector vector;
			for( int i = 0; i < static_cast<int>( count ); i++ ) {
				archive >> vector;
				matrix.AddRow( vector );
			}
			archive >> alpha;
		}
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
