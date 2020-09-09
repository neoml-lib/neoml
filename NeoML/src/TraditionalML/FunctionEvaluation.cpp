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

#include <NeoML/TraditionalML/FunctionEvaluation.h>

namespace NeoML {

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
// IParamTraits
CFunctionParam IParamTraits::GenerateRandom( CRandom&, const CFunctionParam&, const CFunctionParam& ) const
{
	NeoAssert(0);
	return 0;
}

CFunctionParam IParamTraits::Mutate( CRandom&, const CFunctionParam&,
	const CFunctionParam&, const CFunctionParam&, double,
	const CFunctionParam&, const CFunctionParam& ) const
{
	NeoAssert(0);
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
// IFunctionEvaluation
void IFunctionEvaluation::Evaluate( const CArray<CFunctionParamVector>& params, CArray<CFunctionParam>& results )
{
	results.SetSize( params.Size() );

	for( int i = 0; i < params.Size(); ++i ) {
		NeoPresume( params[i].Size() == NumberOfDimensions() );
		results[i] = Evaluate( params[i] );
	}
}

CFunctionParam IFunctionEvaluation::Evaluate( const CFunctionParamVector& params )
{
	NeoPresume( params.Size() == NumberOfDimensions() );

	CArray<CFunctionParam> results;
	CArray<CFunctionParamVector> packedParams;
	packedParams.Add( params );

	Evaluate( packedParams, results );
	NeoPresume( !results.IsEmpty() );

	return results[0];
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
// CDoubleTraits
static CDoubleTraits DoubleTraits;

const CDoubleTraits& CDoubleTraits::GetInstance()
{
	return DoubleTraits;
}

CFunctionParam CDoubleTraits::GenerateRandom( CRandom& random, const CFunctionParam& min, const CFunctionParam& max ) const
{
	return Box( random.Uniform( Unbox( min ), Unbox( max ) ) );
}

CFunctionParam CDoubleTraits::Mutate( CRandom& random, const CFunctionParam& _base,
		const CFunctionParam& _left, const CFunctionParam& _right, double fluctuation,
		const CFunctionParam& _min, const CFunctionParam& _max ) const
{
	double base = Unbox( _base );
	double minVal = Unbox( _min );
	double maxVal = Unbox( _max );

	double muteVal = base + fluctuation * ( Unbox( _left ) - Unbox( _right ) );
	if( muteVal < minVal ) {
		muteVal = minVal + random.Uniform( 0, 1 ) * ( base - minVal );
	} else if( muteVal > maxVal ) {
		muteVal = maxVal - random.Uniform( 0, 1 ) * ( maxVal - base );
	}
	return Box( min( max( muteVal, minVal ), maxVal ) );
}

bool CDoubleTraits::Less( const CFunctionParam& _left, const CFunctionParam& _right ) const
{
	return Unbox( _left ) < Unbox( _right );
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
// CIntTraits
static CIntTraits IntTraits;

const CIntTraits& CIntTraits::GetInstance()
{
	return IntTraits;
}

CFunctionParam CIntTraits::GenerateRandom( CRandom& random, const CFunctionParam& min, const CFunctionParam& max ) const
{
	return Box( random.UniformInt( Unbox( min ), Unbox( max ) ) );
}

CFunctionParam CIntTraits::Mutate( CRandom& random, const CFunctionParam& _base,
		const CFunctionParam& _left, const CFunctionParam& _right, double fluctuation,
		const CFunctionParam& _min, const CFunctionParam& _max ) const
{
	int base = Unbox( _base );
	int minVal = Unbox( _min );
	int maxVal = Unbox( _max );

	int muteVal = base + (int)( fluctuation * ( Unbox( _left ) - Unbox( _right ) ) );
	if( muteVal < minVal ) {
		muteVal = minVal + (int)( random.Uniform( 0, 1 ) * ( base - minVal ) );
	} else if( muteVal > maxVal ) {
		muteVal = maxVal - (int)( random.Uniform( 0, 1 ) * ( maxVal - base ) );
	}
	return Box( min( max( muteVal, minVal ), maxVal ) );
}

bool CIntTraits::Less( const CFunctionParam& _left, const CFunctionParam& _right ) const
{
	return Unbox( _left ) < Unbox( _right );
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
// CFunctionEvaluation
CFunctionEvaluation::CFunctionEvaluation( CFunction& _func )
	: func( _func )
	, minConstraint( _func.NumberOfDimensions() )
	, maxConstraint( _func.NumberOfDimensions() )
{
	int dims = func.NumberOfDimensions();
	CArray<CFunctionParam>& minConstraintArr = minConstraint.CopyOnWrite();
	CArray<CFunctionParam>& maxConstraintArr = maxConstraint.CopyOnWrite();
	const IParamTraits& traits = CDoubleTraits::GetInstance();
	for( int i = 0; i < dims; ++i ) {
		minConstraintArr[i] = traits.GetDefaultMin();
		maxConstraintArr[i] = traits.GetDefaultMax();
	}
}

CFunctionParam CFunctionEvaluation::Evaluate( const CFunctionParamVector& param )
{
	NeoPresume( param.Size() == NumberOfDimensions() );
	CFloatVector vec( param.Size() );
	float* vecPtr = vec.CopyOnWrite();
	for( int i = 0; i < param.Size(); ++i ) {
		vecPtr[i] = static_cast<float>( CDoubleTraits::Unbox( param[i] ) );
	}

	return CDoubleTraits::Box( func.Evaluate( vec ) );
}

}
