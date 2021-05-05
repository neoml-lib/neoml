/* Copyright © 2017-2021 ABBYY Production LLC

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

#include <NeoML/Dnn/AutoDiff.h>
#include <NeoML/Dnn/AutoDiffFunctions.h>

namespace NeoML {

static CPtr<CDnnBlob> callGradient( const CDnnBlob* blob, const CTapeBlob* var )
{
	NeoAssert( var != 0 );

	if( blob == 0 ) {
		return 0;
	}

	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>(blob);
	if( tapeBlob == 0 ) {
		return 0;
	}

	CPtr<IGradientTape> tape = tapeBlob->Tape();
	if( tape == 0 ) {
		return 0;
	}

	CPtr<const ITapeOperation> tapeOperation( tape->GetOperation( tapeBlob ) );
	if( tapeOperation == 0 ) {
		return 0;
	}

	CPtr<CDnnBlob> result = tapeOperation->Gradient( var );
	NeoAssert( result->GetObjectSize() == var->GetDataSize() );
	return result;
}

//------------------------------------------------------------------------------------------------------------

CPtr<const CDnnBlob> Const( IMathEngine& mathEngine, float data, const CBlobDesc& desc )
{
	CPtr<CDnnBlob> result( new CTapeBlob( 0, mathEngine, desc ) );
	result->Fill( data );
	return result.Ptr();
}

CPtr<const CDnnBlob> Const( IMathEngine& mathEngine, float* data, const CBlobDesc& desc )
{
	CPtr<CDnnBlob> result( new CTapeBlob( 0, mathEngine, desc ) );
	result->CopyFrom( data );
	return result.Ptr();
}

CPtr<const CDnnBlob> Const( IMathEngine& mathEngine, const CArray<float>& data, const CBlobDesc& desc )
{
	NeoAssert( desc.BlobSize() == data.Size() );
	CPtr<CDnnBlob> result( new CTapeBlob( 0, mathEngine, desc ) );
	result->CopyFrom( data.GetPtr() );
	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeAdd : public ITapeOperation {
public:
	CTapeAdd( const CDnnBlob& first, const CDnnBlob* second );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
	CPtr<const CDnnBlob> second;
};

CTapeAdd::CTapeAdd( const CDnnBlob& _first, const CDnnBlob* _second ) :
	first( &_first ),
	second( _second )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 || dynamic_cast<const CTapeBlob*>(second.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeAdd::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> firstGrad = callGradient( first, var );
	CPtr<CDnnBlob> secondGrad = callGradient( second, var );
	if( firstGrad == 0 ) {
		return secondGrad;
	}
	if( secondGrad == 0 ) {
		return firstGrad;
	}

	if( firstGrad->GetDataSize() < secondGrad->GetDataSize() ) {
		firstGrad->GetMathEngine().AddDiagMatrixToMatrix( firstGrad->GetData(), secondGrad->GetData(),
			secondGrad->GetObjectCount(), secondGrad->GetObjectSize(), secondGrad->GetData() );
		return secondGrad;
	} else if( secondGrad->GetDataSize() < firstGrad->GetDataSize() ) {
		firstGrad->GetMathEngine().AddDiagMatrixToMatrix( secondGrad->GetData(), firstGrad->GetData(),
			firstGrad->GetObjectCount(), firstGrad->GetObjectSize(), firstGrad->GetData() );
		return firstGrad;
	}

	firstGrad->GetMathEngine().VectorAdd(firstGrad->GetData(), secondGrad->GetData(), firstGrad->GetData(), firstGrad->GetDataSize());
	return firstGrad;
}

CPtr<const CDnnBlob> Add( const CDnnBlob* first, const CDnnBlob* second )
{
	NeoAssert( first != 0 );
	NeoAssert( second != 0 );
	NeoAssert( first->GetDesc().HasEqualDimensions( second->GetDesc() ) );

	IMathEngine& mathEngine = first->GetMathEngine();

	const CTapeBlob* tapeBlob1 = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape1 = tapeBlob1 != 0 ? tapeBlob1->Tape() : 0;
	const CTapeBlob* tapeBlob2 = dynamic_cast<const CTapeBlob*>( second );
	IGradientTape* tape2 = tapeBlob2 != 0 ? tapeBlob2->Tape() : 0;

	assert( tape1 == 0 || tape2 == 0 || tape1 == tape2 );

	IGradientTape* tape = tape1 != 0 ? tape1 : tape2;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, first->GetMathEngine(), first->GetDesc() ) );
	mathEngine.VectorAdd(first->GetData(), second->GetData(), result->GetData(), result->GetDataSize());

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeAdd( *first, second ) ); 
		tape->Add( result, operation );
	}
	return result.Ptr();
}

CPtr<const CDnnBlob> Add( const CDnnBlob* first, float second )
{
	NeoAssert( first != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();

	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CFloatHandleStackVar secondHandle( mathEngine, 1 );
	secondHandle.SetValue( second );
	CPtr<CTapeBlob> result( new CTapeBlob( tape, first->GetMathEngine(), first->GetDesc() ) );
	mathEngine.VectorAddValue(first->GetData(), result->GetData(), result->GetDataSize(), secondHandle );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeAdd( *first, nullptr ) ); 
		tape->Add( result, operation );
	}
	return result.Ptr();
}

CPtr<const CDnnBlob> Add( float first, const CDnnBlob* second )
{
	return Add( second, first );
}

//------------------------------------------------------------------------------------------------------------

class CTapeSub : public ITapeOperation {
public:
	CTapeSub( const CDnnBlob* first, const CDnnBlob* second );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
	CPtr<const CDnnBlob> second;
};

CTapeSub::CTapeSub( const CDnnBlob* _first, const CDnnBlob* _second ) :
	first( _first ),
	second( _second )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 || dynamic_cast<const CTapeBlob*>(second.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeSub::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> firstGrad = callGradient( first, var );
	CPtr<CDnnBlob> secondGrad = callGradient( second, var );

	IMathEngine& mathEngine = first != 0 ? first->GetMathEngine() : second->GetMathEngine() ;

	if( secondGrad != 0 ) {
		mathEngine.VectorNeg( secondGrad->GetData(), secondGrad->GetData(), secondGrad->GetDataSize() );
	}

	if( firstGrad == 0 ) {
		return secondGrad;
	}
	if( secondGrad == 0 ) {
		return firstGrad;
	}

	if( firstGrad->GetDataSize() < secondGrad->GetDataSize() ) {
		firstGrad->GetMathEngine().AddDiagMatrixToMatrix( firstGrad->GetData(), secondGrad->GetData(),
			secondGrad->GetObjectCount(), secondGrad->GetObjectSize(), secondGrad->GetData() );
		return secondGrad;
	} else if( secondGrad->GetDataSize() < firstGrad->GetDataSize() ) {
		firstGrad->GetMathEngine().AddDiagMatrixToMatrix( secondGrad->GetData(), firstGrad->GetData(),
			firstGrad->GetObjectCount(), firstGrad->GetObjectSize(), firstGrad->GetData() );
		return firstGrad;
	}

	firstGrad->GetMathEngine().VectorAdd(firstGrad->GetData(), secondGrad->GetData(), firstGrad->GetData(), firstGrad->GetDataSize());
	return firstGrad;
}

CPtr<const CDnnBlob> Sub( const CDnnBlob* first, const CDnnBlob* second )
{
	NeoAssert( first != 0 );
	NeoAssert( second != 0 );
	NeoAssert( first->GetDesc().HasEqualDimensions( second->GetDesc() ) );

	IMathEngine& mathEngine = first->GetMathEngine();

	const CTapeBlob* tapeBlob1 = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape1 = tapeBlob1 != 0 ? tapeBlob1->Tape() : 0;
	const CTapeBlob* tapeBlob2 = dynamic_cast<const CTapeBlob*>( second );
	IGradientTape* tape2 = tapeBlob2 != 0 ? tapeBlob2->Tape() : 0;

	assert( tape1 == 0 || tape2 == 0 || tape1 == tape2 );

	IGradientTape* tape = tape1 != 0 ? tape1 : tape2;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, first->GetMathEngine(), first->GetDesc() ) );
	mathEngine.VectorSub(first->GetData(), second->GetData(), result->GetData(), result->GetDataSize());

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeSub( first, second ) ); 
		tape->Add( result, operation );
	}
	return result.Ptr();
}

CPtr<const CDnnBlob> Sub( const CDnnBlob* first, float second )
{
	NeoAssert( first != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();

	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, first->GetMathEngine(), first->GetDesc() ) );
	mathEngine.VectorSub( first->GetData(), second, result->GetData(), result->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeSub( first, 0 ) ); 
		tape->Add( result, operation );
	}
	return result.Ptr();
}

CPtr<const CDnnBlob> Sub( float first, const CDnnBlob* second )
{
	NeoAssert( second != 0 );

	IMathEngine& mathEngine = second->GetMathEngine();

	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( second );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, second->GetMathEngine(), second->GetDesc() ) );
	mathEngine.VectorSub( first, second->GetData(), result->GetData(), result->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeSub( 0, second ) ); 
		tape->Add( result, operation );
	}
	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeMult : public ITapeOperation {
public:
	explicit CTapeMult( const CDnnBlob& first, const CDnnBlob& second );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
	CPtr<const CDnnBlob> second;
};

CTapeMult::CTapeMult( const CDnnBlob& _first, const CDnnBlob& _second ) :
	first( &_first ),
	second( &_second )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 || dynamic_cast<const CTapeBlob*>(second.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeMult::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> result;
	CPtr<CDnnBlob> firstGrad = callGradient( first, var );
	CPtr<CDnnBlob> secondGrad = callGradient( second, var );

	if( firstGrad != 0 ) {
		if( firstGrad->GetObjectCount() == 1 ) {
			assert( firstGrad->GetDataSize() == second->GetDataSize() );
			firstGrad->GetMathEngine().VectorEltwiseMultiply( firstGrad->GetData(), second->GetData(), firstGrad->GetData(),
				firstGrad->GetDataSize() );
		} else {
			result = firstGrad->GetClone();
			firstGrad->GetMathEngine().MultiplyDiagMatrixByMatrix( second->GetData(), second->GetDataSize(),
				firstGrad->GetData(), firstGrad->GetObjectSize(), result->GetData(), result->GetDataSize() );
			swap( result, firstGrad );
		}
	}

	if( secondGrad != 0 ) {
		if( secondGrad->GetObjectCount() == 1 ) {
			assert( secondGrad->GetDataSize() == first->GetDataSize() );
			secondGrad->GetMathEngine().VectorEltwiseMultiply( secondGrad->GetData(), first->GetData(), secondGrad->GetData(),
				secondGrad->GetDataSize() );
		} else {
			if( result == 0 ) {
				result = secondGrad->GetClone();
			}
			secondGrad->GetMathEngine().MultiplyDiagMatrixByMatrix( first->GetData(), first->GetDataSize(),
				secondGrad->GetData(), secondGrad->GetObjectSize(), result->GetData(), result->GetDataSize() );
			swap( result, secondGrad );
		}
	}

	if( firstGrad == 0 ) {
		return secondGrad;
	}
	if( secondGrad == 0 ) {
		return firstGrad;
	}

	if( firstGrad->GetDataSize() < secondGrad->GetDataSize() ) {
		firstGrad->GetMathEngine().AddDiagMatrixToMatrix( firstGrad->GetData(), secondGrad->GetData(),
			secondGrad->GetObjectCount(), secondGrad->GetObjectSize(), secondGrad->GetData() );
		return secondGrad;
	} else if( secondGrad->GetDataSize() < firstGrad->GetDataSize() ) {
		firstGrad->GetMathEngine().AddDiagMatrixToMatrix( secondGrad->GetData(), firstGrad->GetData(),
			firstGrad->GetObjectCount(), firstGrad->GetObjectSize(), firstGrad->GetData() );
		return firstGrad;
	}

	firstGrad->GetMathEngine().VectorAdd(firstGrad->GetData(), secondGrad->GetData(), firstGrad->GetData(), firstGrad->GetDataSize());
	return firstGrad;
}

CPtr<const CDnnBlob> Mult( const CDnnBlob* first, const CDnnBlob* second )
{
	NeoAssert( first != 0 );
	NeoAssert( second != 0 );
	NeoAssert( first->GetDesc().HasEqualDimensions( second->GetDesc() ) );

	IMathEngine& mathEngine = first->GetMathEngine();

	const CTapeBlob* tapeBlob1 = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape1 = tapeBlob1 != 0 ? tapeBlob1->Tape() : 0;
	const CTapeBlob* tapeBlob2 = dynamic_cast<const CTapeBlob*>( second );
	IGradientTape* tape2 = tapeBlob2 != 0 ? tapeBlob2->Tape() : 0;

	assert( tape1 == 0 || tape2 == 0 || tape1 == tape2 );

	IGradientTape* tape = tape1 != 0 ? tape1 : tape2;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, first->GetDesc() ) );
	mathEngine.VectorEltwiseMultiply( first->GetData(), second->GetData(), result->GetData(), result->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeMult( *first, *second ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

CPtr<const CDnnBlob> Mult( const CDnnBlob* first, float value )
{
	NeoAssert( first != 0 );

	CPtr<const CDnnBlob> second = Const( first->GetMathEngine(), value, first->GetDesc() );
	return Mult( first, second );
}

CPtr<const CDnnBlob> Mult( float first, const CDnnBlob* second )
{
	return Mult( second, first );
}

//------------------------------------------------------------------------------------------------------------

class CTapeDiv : public ITapeOperation {
public:
	explicit CTapeDiv( const CDnnBlob& first, const CDnnBlob& second );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
	CPtr<const CDnnBlob> second;
};

CTapeDiv::CTapeDiv( const CDnnBlob& _first, const CDnnBlob& _second ) :
	first( &_first ),
	second( &_second )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 || dynamic_cast<const CTapeBlob*>(second.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeDiv::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> result;
	CPtr<CDnnBlob> firstGrad = callGradient( first, var );
	CPtr<CDnnBlob> secondGrad = callGradient( second, var );

	if( firstGrad == 0 && secondGrad == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = first->GetMathEngine();
	const int gradientSize = firstGrad != 0 ? firstGrad->GetObjectSize() : secondGrad->GetObjectSize();
	const int vectorSize = first->GetDataSize();

	if( firstGrad == 0 ) {
		if( secondGrad->GetObjectCount() == 1 ) {
			mathEngine.VectorEltwiseDivide( secondGrad->GetData(), first->GetData(), secondGrad->GetData(),
				secondGrad->GetDataSize() );
		} else {
			mathEngine.MatrixColumnsEltwiseDivide( secondGrad->GetData(), secondGrad->GetObjectCount(), gradientSize,
				first->GetData(), secondGrad->GetData() );
		}
		return secondGrad;
	}
	
	if( secondGrad == 0 ) {
		if( firstGrad->GetObjectCount() == 1 ) {
			mathEngine.VectorEltwiseDivide( firstGrad->GetData(), second->GetData(), firstGrad->GetData(),
				firstGrad->GetDataSize() );
		} else {
			mathEngine.MatrixColumnsEltwiseDivide( firstGrad->GetData(), firstGrad->GetObjectCount(), gradientSize,
				second->GetData(), firstGrad->GetData() );
		}
		return firstGrad;
	}

	if( firstGrad->GetObjectCount() == 1 ) {
		assert( firstGrad->GetDataSize() == second->GetDataSize() );
		mathEngine.VectorEltwiseMultiply( firstGrad->GetData(), second->GetData(), firstGrad->GetData(),
			firstGrad->GetDataSize() );
	} else {
		result = firstGrad->GetClone();
		mathEngine.MultiplyDiagMatrixByMatrix( second->GetData(), second->GetDataSize(),
			firstGrad->GetData(), firstGrad->GetObjectSize(), result->GetData(), result->GetDataSize() );
		swap( result, firstGrad );
	}

	if( secondGrad->GetObjectCount() == 1 ) {
		assert( secondGrad->GetDataSize() == first->GetDataSize() );
		mathEngine.VectorEltwiseMultiply( secondGrad->GetData(), first->GetData(), secondGrad->GetData(),
			secondGrad->GetDataSize() );
		secondGrad->GetMathEngine().VectorNeg( secondGrad->GetData(), secondGrad->GetData(), secondGrad->GetDataSize() );
	} else {
		if( result == 0 ) {
			result = secondGrad->GetClone();
		}
		mathEngine.MultiplyDiagMatrixByMatrix( first->GetData(), first->GetDataSize(),
			secondGrad->GetData(), secondGrad->GetObjectSize(), result->GetData(), result->GetDataSize() );
		secondGrad->GetMathEngine().VectorNeg( result->GetData(), secondGrad->GetData(), result->GetDataSize() );
	}

	CFloatHandleStackVar secondSquare( mathEngine, second->GetDataSize() );
	mathEngine.VectorEltwiseMultiply( second->GetData(), second->GetData(), secondSquare, gradientSize );

	if( firstGrad->GetDataSize() < secondGrad->GetDataSize() ) {
		mathEngine.AddDiagMatrixToMatrix( firstGrad->GetData(), secondGrad->GetData(),
			secondGrad->GetObjectCount(), secondGrad->GetObjectSize(), secondGrad->GetData() );
		mathEngine.MatrixColumnsEltwiseDivide( secondGrad->GetData(), secondGrad->GetObjectCount(), gradientSize,
			secondSquare.GetHandle(), secondGrad->GetData() );
		return secondGrad;
	} else if( secondGrad->GetDataSize() < firstGrad->GetDataSize() ) {
		mathEngine.AddDiagMatrixToMatrix( secondGrad->GetData(), firstGrad->GetData(),
			firstGrad->GetObjectCount(), firstGrad->GetObjectSize(), firstGrad->GetData() );
		mathEngine.MatrixColumnsEltwiseDivide( firstGrad->GetData(), firstGrad->GetObjectCount(), gradientSize,
			secondSquare.GetHandle(), firstGrad->GetData() );
		return firstGrad;
	}

	mathEngine.VectorAdd(firstGrad->GetData(), secondGrad->GetData(), firstGrad->GetData(), firstGrad->GetDataSize());
	if( firstGrad->GetObjectCount() == 1 ) {
		mathEngine.VectorEltwiseDivide( firstGrad->GetData(), secondSquare.GetHandle(), firstGrad->GetData(), vectorSize );
	} else {
		mathEngine.MatrixColumnsEltwiseDivide( firstGrad->GetData(), vectorSize, gradientSize,
			secondSquare.GetHandle(), firstGrad->GetData() );
	}
	return firstGrad;
}

CPtr<const CDnnBlob> Div( const CDnnBlob* first, const CDnnBlob* second )
{
	NeoAssert( first != 0 );
	NeoAssert( second != 0 );
	NeoAssert( first->GetDesc().HasEqualDimensions( second->GetDesc() ) );

	IMathEngine& mathEngine = first->GetMathEngine();

	const CTapeBlob* tapeBlob1 = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape1 = tapeBlob1 != 0 ? tapeBlob1->Tape() : 0;
	const CTapeBlob* tapeBlob2 = dynamic_cast<const CTapeBlob*>( second );
	IGradientTape* tape2 = tapeBlob2 != 0 ? tapeBlob2->Tape() : 0;

	assert( tape1 == 0 || tape2 == 0 || tape1 == tape2 );

	IGradientTape* tape = tape1 != 0 ? tape1 : tape2;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, first->GetDesc() ) );
	mathEngine.VectorEltwiseDivide( first->GetData(), second->GetData(), result->GetData(), result->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeDiv( *first, *second ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

CPtr<const CDnnBlob> Div( const CDnnBlob* first, float value )
{
	NeoAssert( first != 0 );
	CPtr<const CDnnBlob> second = Const( first->GetMathEngine(), value, first->GetDesc() );
	return Div( first, second );
}

CPtr<const CDnnBlob> NEOML_API Div( float value, const CDnnBlob* second )
{
	NeoAssert( second != 0 );
	CPtr<const CDnnBlob> first = Const( second->GetMathEngine(), value, second->GetDesc() );
	return Div( first, second );
}

//------------------------------------------------------------------------------------------------------------

class CTapeMax : public ITapeOperation {
public:
	explicit CTapeMax( const CDnnBlob& first, float second );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
	const float second;
};

CTapeMax::CTapeMax( const CDnnBlob& _first, float _second ) :
	first( &_first ),
	second( _second )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeMax::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> grad = callGradient( first, var );
	if( grad == 0 ) {
		return 0;
	}

	grad->GetMathEngine().VectorMaxDiff( first->GetData(), second, grad->GetData(),
		grad->GetObjectCount(), grad->GetObjectSize() );
	return grad;
}

CPtr<const CDnnBlob> NEOML_API Max( const CDnnBlob* first, float second )
{
	NeoAssert( first != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();

	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, first->GetDesc() ) );
	mathEngine.VectorMax( first->GetData(), second, result->GetData(), result->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeMax( *first, second ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

CPtr<const CDnnBlob> NEOML_API Max( float first, const CDnnBlob* second )
{
	return Max( second, first );
}

//------------------------------------------------------------------------------------------------------------

class CTapeSum : public ITapeOperation {
public:
	explicit CTapeSum( const CDnnBlob& first );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
};

CTapeSum::CTapeSum( const CDnnBlob& _first ) :
	first( &_first )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeSum::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> grad = callGradient( first, var );
	if( grad == 0 ) {
		return 0;
	}
	int height = grad->GetObjectCount();
	int width = grad->GetObjectSize();

	if( height == 1 ) {
		return grad;
	}

	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( grad->GetMathEngine(), { width } );
	result->GetMathEngine().SumMatrixColumns( result->GetData(), grad->GetData(), height, width );
	return result;
}

CPtr<const CDnnBlob> Sum( const CDnnBlob* first )
{
	NeoAssert( first != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();
	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, CBlobDesc( {1} ) ) );
	mathEngine.VectorSum( first->GetData(), first->GetDataSize(), result->GetData() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeSum( *tapeBlob ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeNeg : public ITapeOperation {
public:
	explicit CTapeNeg( const CDnnBlob& first );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
};

CTapeNeg::CTapeNeg( const CDnnBlob& _first ) :
	first( &_first )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeNeg::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> grad = callGradient( first, var );
	if( grad == 0 ) {
		return 0;
	}

	grad->GetMathEngine().VectorNeg( grad->GetData(), grad->GetData(), grad->GetDataSize() );
	return grad;
}

CPtr<const CDnnBlob> Neg( const CDnnBlob* first )
{
	NeoAssert( first != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();
	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, first->GetDesc() ) );
	mathEngine.VectorNeg( first->GetData(), result->GetData(), first->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeNeg( *tapeBlob ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeAbs : public ITapeOperation {
public:
	explicit CTapeAbs( const CDnnBlob& first );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
};

CTapeAbs::CTapeAbs( const CDnnBlob& _first ) :
	first( &_first )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeAbs::Gradient( const CTapeBlob* var ) const
{
	return callGradient( first, var );
}

CPtr<const CDnnBlob> Abs( const CDnnBlob* first )
{
	NeoAssert( first != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();
	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, first->GetDesc() ) );
	mathEngine.VectorAbs( first->GetData(), result->GetData(), first->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeAbs( *tapeBlob ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeExp : public ITapeOperation {
public:
	explicit CTapeExp( const CDnnBlob& first );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
};

CTapeExp::CTapeExp( const CDnnBlob& _first ) :
	first( &_first )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeExp::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> grad = callGradient( first, var );
	if( grad == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = first->GetMathEngine();
	CFloatHandleStackVar expV( mathEngine, first->GetDataSize() );
	mathEngine.VectorExp( first->GetData(), expV, first->GetDataSize() );

	if( grad->GetObjectCount() == 1 ) {
		NeoAssert( grad->GetDataSize() == first->GetDataSize() );
		mathEngine.VectorEltwiseMultiply( grad->GetData(), expV, grad->GetData(), grad->GetDataSize() );
		return grad;
	}

	CPtr<CDnnBlob> result = grad->GetClone();
	mathEngine.MultiplyDiagMatrixByMatrix( expV, first->GetDataSize(), grad->GetData(), grad->GetObjectSize(),
		result->GetData(), result->GetDataSize() );
	return result;
}

CPtr<const CDnnBlob> Exp( const CDnnBlob* first )
{
	NeoAssert( first != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();
	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, first->GetDesc() ) );
	mathEngine.VectorExp( first->GetData(), result->GetData(), first->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeExp( *tapeBlob ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeLog : public ITapeOperation {
public:
	explicit CTapeLog( const CDnnBlob& first );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
};

CTapeLog::CTapeLog( const CDnnBlob& _first ) :
	first( &_first )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeLog::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> grad = callGradient( first, var );
	if( grad == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = first->GetMathEngine();

	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, grad->GetDesc() );
	mathEngine.VectorLogDiff( grad->GetData(), grad->GetObjectCount(), grad->GetObjectSize(), first->GetData(), result->GetData() );
	return result;
}

CPtr<const CDnnBlob> Log( const CDnnBlob* first )
{
	NeoAssert( first != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();
	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, first->GetDesc() ) );
	mathEngine.VectorLog( first->GetData(), result->GetData(), first->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeLog( *tapeBlob ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeTopK : public ITapeOperation {
public:
	explicit CTapeTopK( const CDnnBlob& first, const CDnnBlob& indices );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
	CPtr<const CDnnBlob> indices;
};

CTapeTopK::CTapeTopK( const CDnnBlob& _first, const CDnnBlob& _indices ) :
	first( &_first ),
	indices( &_indices )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeTopK::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> grad = callGradient( first, var );
	if( grad == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = first->GetMathEngine();

	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, {indices->GetDataSize(), 1, 1, 1, 1, 1, grad->GetObjectSize()} );
	mathEngine.VectorTopKDiff( grad->GetData(), grad->GetObjectCount(), grad->GetObjectSize(), indices->GetData<int>(),
		indices->GetDataSize(), result->GetData() );
	return result;
}

CPtr<const CDnnBlob> NEOML_API TopK( const CDnnBlob* first, int k )
{
	NeoAssert( first != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();
	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, {k} ) );
	CPtr<CDnnBlob> indices( CDnnBlob::CreateBlob( mathEngine, CT_Int, {k} ) );

	mathEngine.VectorTopK( first->GetData(), first->GetDataSize(), k, result->GetData(), indices->GetData<int>() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeTopK( *tapeBlob, *indices ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeClip : public ITapeOperation {
public:
	explicit CTapeClip( const CDnnBlob& first );

	CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
};

CTapeClip::CTapeClip( const CDnnBlob& _first ) :
	first( &_first )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeClip::Gradient( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> grad = callGradient( first, var );
	return grad;
}

CPtr<const CDnnBlob> Clip( const CDnnBlob* first, float minValue, float maxValue )
{
	NeoAssert( first != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();
	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	CFloatHandleStackVar minHandle( mathEngine, 1 );
	minHandle.SetValue( minValue );
	CFloatHandleStackVar maxHandle( mathEngine, 1 );
	maxHandle.SetValue( maxValue );

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, first->GetDesc() ) );
	mathEngine.VectorMinMax( first->GetData(), result->GetData(), first->GetDataSize(), minHandle, maxHandle );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeClip( *tapeBlob ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

CPtr<const CDnnBlob> BinaryCrossEntropy( const CDnnBlob* labels, const CDnnBlob* preds, bool fromLogits )
{
	NeoAssert( labels != 0 );
	NeoAssert( preds != 0 );
	NeoAssert( labels->GetDesc().HasEqualDimensions( preds->GetDesc() ) );

	IMathEngine& mathEngine = preds->GetMathEngine();
	const int vectorSize = preds->GetDataSize();

	// Notations:
	// x = logits, z = labels

	// The original loss function formula:
	// loss = (1 - z) * x + log(1 + exp(-x))

	// The formula to avoid overflow for large exponent power in exp(-x):
	// loss = (1 - z) * x + log(1 + exp(-abs(x))) + max(-x, 0)

	CPtr<const CDnnBlob> clippedPreds = fromLogits ? preds : Clip( preds, 0.0000001f, 0.9999999f );

	CPtr<const CDnnBlob> x = fromLogits ? clippedPreds : Log( Div( clippedPreds, Sub(1, clippedPreds) ) );
	CPtr<const CDnnBlob> temp1 = Mult( Sub( 1, labels ), x );
	CPtr<const CDnnBlob> temp2 = Log( Add( 1, Exp( Neg( Abs(x) ) ) ) );
	CPtr<const CDnnBlob> temp3 = Max( Neg(x), 0 );

	return Add( Add( temp1, temp2 ), temp3 );
}

} // namespace NeoML
