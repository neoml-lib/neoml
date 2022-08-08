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

static CPtr<CDnnBlob> callJacobian( const CDnnBlob* blob, const CTapeBlob* var )
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

	CPtr<CDnnBlob> result = tapeOperation->Jacobian( var );
	NeoAssert( result->GetObjectSize() == var->GetDataSize() );
	return result;
}

static bool isSequentialAxes( const CDnnBlob* blob, const CArray<int>& axes )
{
	for( int i = 1; i < axes.Size(); i++ ) {
		NeoAssert( axes[i - 1] < axes[i] );
		for( int j = axes[i - 1] + 1; j < axes[i]; j++ ) {
			if( blob->DimSize( j ) != 1 ) {
				return false;
			}
		}
	}
	return true;
}

static void getSequentialAxesDimensions( const CDnnBlob* first, const CArray<int>& axes, int& followingDimension,
	int& dimension, int& precedingDimension )
{
	NeoPresume( isSequentialAxes( first, axes ) );

	followingDimension = 1;
	for( int d = 0; d < axes.First(); d++ ) {
		followingDimension *= first->DimSize( d );
	}
	dimension = 1;
	for( int d = axes.First(); d <= axes.Last(); d++ ) {
		dimension *= first->DimSize( d );
	}
	precedingDimension = 1;
	for( int d = axes.Last() + 1; d < BD_Count; d++ ) {
		precedingDimension *= first->DimSize( d );
	}
}

static CPtr<CDnnBlob> diagJacobianToFull( const CPtr<CDnnBlob>& diag )
{
	IMathEngine& mathEngine = diag->GetMathEngine();
	const int width = diag->GetObjectSize();
	CPtr<CDnnBlob> full = CDnnBlob::CreateBlob( mathEngine, { width, 1, 1, 1, 1, 1, width } );
	mathEngine.VectorFill( full->GetData(), 0.f, width * width );
	mathEngine.AddDiagMatrixToMatrix( diag->GetData(), full->GetData(),
		width, width, full->GetData() );
	return full;
}

static CBlobDesc getBroadcastedDesc( const CBlobDesc& firstDesc, const CBlobDesc& secondDesc )
{
	CBlobDesc res( firstDesc.GetDataType() );
	for( int i = 0; i < BD_Count; i++ ) {
		const int firstDim = firstDesc.DimSize( i );
		const int secondDim = secondDesc.DimSize( i );
		NeoAssert( firstDim == secondDim || firstDim == 1 || secondDim == 1 );
		res.SetDimSize( i, max( firstDim, secondDim ) );
	}
	return res;
}

//------------------------------------------------------------------------------------------------------------

CPtr<const CDnnBlob> Const( IMathEngine& mathEngine, float data, const CBlobDesc& desc )
{
	CPtr<CDnnBlob> result( new CTapeBlob( 0, mathEngine, desc ) );
	result->Fill( data );
	return result.Ptr();
}

CPtr<const CDnnBlob> Const( IMathEngine& mathEngine, const float* data, const CBlobDesc& desc )
{
	CPtr<CDnnBlob> result( new CTapeBlob( 0, mathEngine, desc ) );
	result->CopyFrom( data );
	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeAdd : public ITapeOperation {
public:
	CTapeAdd( const CDnnBlob* first, const CDnnBlob* second );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
	CPtr<const CDnnBlob> second;
};

CTapeAdd::CTapeAdd( const CDnnBlob* _first, const CDnnBlob* _second ) :
	first( _first ),
	second( _second )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 || dynamic_cast<const CTapeBlob*>(second.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeAdd::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> firstJacobian = callJacobian( first, var );
	CPtr<CDnnBlob> secondJacobian = callJacobian( second, var );

	if( firstJacobian == 0 ) {
		return secondJacobian;
	}
	if( secondJacobian == 0 ) {
		return firstJacobian;
	}

	if( firstJacobian->GetDataSize() < secondJacobian->GetDataSize() ) {
		firstJacobian->GetMathEngine().AddDiagMatrixToMatrix( firstJacobian->GetData(), secondJacobian->GetData(),
			secondJacobian->GetObjectCount(), secondJacobian->GetObjectSize(), secondJacobian->GetData() );
		return secondJacobian;
	} else if( secondJacobian->GetDataSize() < firstJacobian->GetDataSize() ) {
		firstJacobian->GetMathEngine().AddDiagMatrixToMatrix( secondJacobian->GetData(), firstJacobian->GetData(),
			firstJacobian->GetObjectCount(), firstJacobian->GetObjectSize(), firstJacobian->GetData() );
		return firstJacobian;
	}

	firstJacobian->GetMathEngine().VectorAdd(firstJacobian->GetData(), secondJacobian->GetData(), firstJacobian->GetData(), firstJacobian->GetDataSize());
	return firstJacobian;
}

CPtr<const CDnnBlob> Add( const CDnnBlob* first, const CDnnBlob* second )
{
	NeoAssert( first != 0 );
	NeoAssert( second != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();

	CBlobDesc broadcastedDesc = getBroadcastedDesc( first->GetDesc(), second->GetDesc() );
	CPtr<const CDnnBlob> firstBlob = Broadcast( first, broadcastedDesc );
	CPtr<const CDnnBlob> secondBlob = Broadcast( second, broadcastedDesc );

	NeoAssert( firstBlob->GetDesc().HasEqualDimensions( secondBlob->GetDesc() ) );

	const CTapeBlob* tapeBlob1 = dynamic_cast<const CTapeBlob*>( firstBlob.Ptr() );
	IGradientTape* tape1 = tapeBlob1 != 0 ? tapeBlob1->Tape() : 0;
	const CTapeBlob* tapeBlob2 = dynamic_cast<const CTapeBlob*>( secondBlob.Ptr() );
	IGradientTape* tape2 = tapeBlob2 != 0 ? tapeBlob2->Tape() : 0;

	NeoAssert( tape1 == 0 || tape2 == 0 || tape1 == tape2 );

	IGradientTape* tape = tape1 != 0 ? tape1 : tape2;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, firstBlob->GetMathEngine(), firstBlob->GetDesc() ) );
	mathEngine.VectorAdd( firstBlob->GetData(), secondBlob->GetData(), result->GetData(), result->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeAdd( firstBlob, secondBlob ) ); 
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
		CPtr<ITapeOperation> operation( new CTapeAdd( first, nullptr ) ); 
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

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

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

CPtr<CDnnBlob> CTapeSub::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> firstJacobian = callJacobian( first, var );
	CPtr<CDnnBlob> secondJacobian = callJacobian( second, var );

	IMathEngine& mathEngine = first != 0 ? first->GetMathEngine() : second->GetMathEngine() ;

	if( secondJacobian != 0 ) {
		mathEngine.VectorNeg( secondJacobian->GetData(), secondJacobian->GetData(), secondJacobian->GetDataSize() );
	}

	if( firstJacobian == 0 ) {
		return secondJacobian;
	}
	if( secondJacobian == 0 ) {
		return firstJacobian;
	}

	if( firstJacobian->GetDataSize() < secondJacobian->GetDataSize() ) {
		firstJacobian->GetMathEngine().AddDiagMatrixToMatrix( firstJacobian->GetData(), secondJacobian->GetData(),
			secondJacobian->GetObjectCount(), secondJacobian->GetObjectSize(), secondJacobian->GetData() );
		return secondJacobian;
	} else if( secondJacobian->GetDataSize() < firstJacobian->GetDataSize() ) {
		firstJacobian->GetMathEngine().AddDiagMatrixToMatrix( secondJacobian->GetData(), firstJacobian->GetData(),
			firstJacobian->GetObjectCount(), firstJacobian->GetObjectSize(), firstJacobian->GetData() );
		return firstJacobian;
	}

	firstJacobian->GetMathEngine().VectorAdd(firstJacobian->GetData(), secondJacobian->GetData(), firstJacobian->GetData(), firstJacobian->GetDataSize());
	return firstJacobian;
}

CPtr<const CDnnBlob> Sub( const CDnnBlob* first, const CDnnBlob* second )
{
	NeoAssert( first != 0 );
	NeoAssert( second != 0 );

	CBlobDesc broadcastedDesc = getBroadcastedDesc( first->GetDesc(), second->GetDesc() );
	CPtr<const CDnnBlob> firstBlob = Broadcast( first, broadcastedDesc );
	CPtr<const CDnnBlob> secondBlob = Broadcast( second, broadcastedDesc );

	NeoAssert( firstBlob->GetDesc().HasEqualDimensions( secondBlob->GetDesc() ) );

	IMathEngine& mathEngine = first->GetMathEngine();

	const CTapeBlob* tapeBlob1 = dynamic_cast<const CTapeBlob*>( firstBlob.Ptr() );
	IGradientTape* tape1 = tapeBlob1 != 0 ? tapeBlob1->Tape() : 0;
	const CTapeBlob* tapeBlob2 = dynamic_cast<const CTapeBlob*>( secondBlob.Ptr() );
	IGradientTape* tape2 = tapeBlob2 != 0 ? tapeBlob2->Tape() : 0;

	NeoAssert( tape1 == 0 || tape2 == 0 || tape1 == tape2 );

	IGradientTape* tape = tape1 != 0 ? tape1 : tape2;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, firstBlob->GetMathEngine(), firstBlob->GetDesc() ) );
	mathEngine.VectorSub( firstBlob->GetData(), secondBlob->GetData(), result->GetData(), result->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeSub( firstBlob, secondBlob ) ); 
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

class CTapeMul : public ITapeOperation {
public:
	explicit CTapeMul( const CDnnBlob* first, const CDnnBlob* second );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
	CPtr<const CDnnBlob> second;
};

CTapeMul::CTapeMul( const CDnnBlob* _first, const CDnnBlob* _second ) :
	first( _first ),
	second( _second )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 || dynamic_cast<const CTapeBlob*>(second.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeMul::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> result;
	CPtr<CDnnBlob> firstJacobian = callJacobian( first, var );
	CPtr<CDnnBlob> secondJacobian = callJacobian( second, var );

	if( firstJacobian != 0 ) {
		if( firstJacobian->GetObjectCount() == 1 ) {
			NeoAssert( firstJacobian->GetDataSize() == second->GetDataSize() );
			firstJacobian->GetMathEngine().VectorEltwiseMultiply( firstJacobian->GetData(), second->GetData(), firstJacobian->GetData(),
				firstJacobian->GetDataSize() );
		} else {
			result = firstJacobian->GetClone();
			firstJacobian->GetMathEngine().MultiplyDiagMatrixByMatrix( second->GetData(), second->GetDataSize(),
				firstJacobian->GetData(), firstJacobian->GetObjectSize(), result->GetData(), result->GetDataSize() );
			swap( result, firstJacobian );
		}
	}

	if( secondJacobian != 0 ) {
		if( secondJacobian->GetObjectCount() == 1 ) {
			NeoAssert( secondJacobian->GetDataSize() == first->GetDataSize() );
			secondJacobian->GetMathEngine().VectorEltwiseMultiply( secondJacobian->GetData(), first->GetData(), secondJacobian->GetData(),
				secondJacobian->GetDataSize() );
		} else {
			if( result == 0 ) {
				result = secondJacobian->GetClone();
			}
			secondJacobian->GetMathEngine().MultiplyDiagMatrixByMatrix( first->GetData(), first->GetDataSize(),
				secondJacobian->GetData(), secondJacobian->GetObjectSize(), result->GetData(), result->GetDataSize() );
			swap( result, secondJacobian );
		}
	}

	if( firstJacobian == 0 ) {
		return secondJacobian;
	}
	if( secondJacobian == 0 ) {
		return firstJacobian;
	}

	if( firstJacobian->GetDataSize() < secondJacobian->GetDataSize() ) {
		firstJacobian->GetMathEngine().AddDiagMatrixToMatrix( firstJacobian->GetData(), secondJacobian->GetData(),
			secondJacobian->GetObjectCount(), secondJacobian->GetObjectSize(), secondJacobian->GetData() );
		return secondJacobian;
	} else if( secondJacobian->GetDataSize() < firstJacobian->GetDataSize() ) {
		firstJacobian->GetMathEngine().AddDiagMatrixToMatrix( secondJacobian->GetData(), firstJacobian->GetData(),
			firstJacobian->GetObjectCount(), firstJacobian->GetObjectSize(), firstJacobian->GetData() );
		return firstJacobian;
	}

	firstJacobian->GetMathEngine().VectorAdd(firstJacobian->GetData(), secondJacobian->GetData(), firstJacobian->GetData(), firstJacobian->GetDataSize());
	return firstJacobian;
}

CPtr<const CDnnBlob> Mul( const CDnnBlob* first, const CDnnBlob* second )
{
	NeoAssert( first != 0 );
	NeoAssert( second != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();

	CBlobDesc broadcastedDesc = getBroadcastedDesc( first->GetDesc(), second->GetDesc() );
	CPtr<const CDnnBlob> firstBlob = Broadcast( first, broadcastedDesc );
	CPtr<const CDnnBlob> secondBlob = Broadcast( second, broadcastedDesc );

	NeoAssert( firstBlob->GetDesc().HasEqualDimensions( secondBlob->GetDesc() ) );

	const CTapeBlob* tapeBlob1 = dynamic_cast<const CTapeBlob*>( firstBlob.Ptr() );
	IGradientTape* tape1 = tapeBlob1 != 0 ? tapeBlob1->Tape() : 0;
	const CTapeBlob* tapeBlob2 = dynamic_cast<const CTapeBlob*>( secondBlob.Ptr() );
	IGradientTape* tape2 = tapeBlob2 != 0 ? tapeBlob2->Tape() : 0;

	NeoAssert( tape1 == 0 || tape2 == 0 || tape1 == tape2 );

	IGradientTape* tape = tape1 != 0 ? tape1 : tape2;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, firstBlob->GetDesc() ) );
	mathEngine.VectorEltwiseMultiply( firstBlob->GetData(), secondBlob->GetData(), result->GetData(), result->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeMul( firstBlob, secondBlob ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

CPtr<const CDnnBlob> Mul( const CDnnBlob* first, float value )
{
	NeoAssert( first != 0 );

	CPtr<const CDnnBlob> second = Const( first->GetMathEngine(), value, first->GetDesc() );
	return Mul( first, second );
}

CPtr<const CDnnBlob> Mul( float first, const CDnnBlob* second )
{
	return Mul( second, first );
}

//------------------------------------------------------------------------------------------------------------

class CTapeDiv : public ITapeOperation {
public:
	explicit CTapeDiv( const CDnnBlob* first, const CDnnBlob* second );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
	CPtr<const CDnnBlob> second;
};

CTapeDiv::CTapeDiv( const CDnnBlob* _first, const CDnnBlob* _second ) :
	first( _first ),
	second( _second )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 || dynamic_cast<const CTapeBlob*>(second.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeDiv::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> result;
	CPtr<CDnnBlob> firstJacobian = callJacobian( first, var );
	CPtr<CDnnBlob> secondJacobian = callJacobian( second, var );

	if( firstJacobian == 0 && secondJacobian == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = first->GetMathEngine();
	const int gradientSize = firstJacobian != 0 ? firstJacobian->GetObjectSize() : secondJacobian->GetObjectSize();
	const int vectorSize = first->GetDataSize();

	if( secondJacobian == 0 ) {
		if( firstJacobian->GetObjectCount() == 1 ) {
			mathEngine.VectorEltwiseDivide( firstJacobian->GetData(), second->GetData(), firstJacobian->GetData(),
				firstJacobian->GetDataSize() );
		} else {
			mathEngine.MatrixColumnsEltwiseDivide( firstJacobian->GetData(), firstJacobian->GetObjectCount(), gradientSize,
				second->GetData(), firstJacobian->GetData() );
		}
		return firstJacobian;
	}

	// firstJacobian = first' * second
	if( firstJacobian != 0 ) {
		if( firstJacobian->GetObjectCount() == 1 ) {
			NeoAssert( firstJacobian->GetDataSize() == second->GetDataSize() );
			mathEngine.VectorEltwiseMultiply( firstJacobian->GetData(), second->GetData(), firstJacobian->GetData(),
				firstJacobian->GetDataSize() );
		} else {
			result = firstJacobian->GetClone();
			mathEngine.MultiplyDiagMatrixByMatrix( second->GetData(), second->GetDataSize(),
				firstJacobian->GetData(), firstJacobian->GetObjectSize(), result->GetData(), result->GetDataSize() );
			swap( result, firstJacobian );
		}
	}

	// secondJacobian = -second' * first
	if( secondJacobian->GetObjectCount() == 1 ) {
		NeoAssert( secondJacobian->GetDataSize() == first->GetDataSize() );
		mathEngine.VectorEltwiseMultiply( secondJacobian->GetData(), first->GetData(), secondJacobian->GetData(),
			secondJacobian->GetDataSize() );
		secondJacobian->GetMathEngine().VectorNeg( secondJacobian->GetData(), secondJacobian->GetData(), secondJacobian->GetDataSize() );
	} else {
		if( result == 0 ) {
			result = secondJacobian->GetClone();
		}
		mathEngine.MultiplyDiagMatrixByMatrix( first->GetData(), first->GetDataSize(),
			secondJacobian->GetData(), secondJacobian->GetObjectSize(), result->GetData(), result->GetDataSize() );
		secondJacobian->GetMathEngine().VectorNeg( result->GetData(), secondJacobian->GetData(), result->GetDataSize() );
	}

	// secondSquare = second * second
	CFloatHandleStackVar secondSquare( mathEngine, second->GetDataSize() );
	mathEngine.VectorEltwiseMultiply( second->GetData(), second->GetData(), secondSquare, second->GetDataSize() );

	if( firstJacobian != 0 ) {
		if( firstJacobian->GetDataSize() < secondJacobian->GetDataSize() ) {
			// secondJacobian = firstJacobian + secondJacobian / secondSquare
			mathEngine.AddDiagMatrixToMatrix( firstJacobian->GetData(), secondJacobian->GetData(),
				secondJacobian->GetObjectCount(), secondJacobian->GetObjectSize(), secondJacobian->GetData() );
			mathEngine.MatrixColumnsEltwiseDivide( secondJacobian->GetData(), secondJacobian->GetObjectCount(), gradientSize,
				secondSquare.GetHandle(), secondJacobian->GetData() );
			return secondJacobian;
		} else if( secondJacobian->GetDataSize() < firstJacobian->GetDataSize() ) {
			// firstJacobian = firstJacobian + secondJacobian / secondSquare
			mathEngine.AddDiagMatrixToMatrix( secondJacobian->GetData(), firstJacobian->GetData(),
				firstJacobian->GetObjectCount(), firstJacobian->GetObjectSize(), firstJacobian->GetData() );
			mathEngine.MatrixColumnsEltwiseDivide( firstJacobian->GetData(), firstJacobian->GetObjectCount(), gradientSize,
				secondSquare.GetHandle(), firstJacobian->GetData() );
			return firstJacobian;
		}
		// secondJacobian = firstJacobian + secondJacobian
		mathEngine.VectorAdd(firstJacobian->GetData(), secondJacobian->GetData(), secondJacobian->GetData(), secondJacobian->GetDataSize());
	}

	// secondJacobian = secondJacobian / secondSquare
	if( secondJacobian->GetObjectCount() == 1 ) {
		mathEngine.VectorEltwiseDivide( secondJacobian->GetData(), secondSquare.GetHandle(), secondJacobian->GetData(), vectorSize );
	} else {
		mathEngine.MatrixColumnsEltwiseDivide( secondJacobian->GetData(), vectorSize, gradientSize,
			secondSquare.GetHandle(), secondJacobian->GetData() );
	}

	return secondJacobian;
}

CPtr<const CDnnBlob> Div( const CDnnBlob* first, const CDnnBlob* second )
{
	NeoAssert( first != 0 );
	NeoAssert( second != 0 );

	CBlobDesc broadcastedDesc = getBroadcastedDesc( first->GetDesc(), second->GetDesc() );
	CPtr<const CDnnBlob> firstBlob = Broadcast( first, broadcastedDesc );
	CPtr<const CDnnBlob> secondBlob = Broadcast( second, broadcastedDesc );

	NeoAssert( firstBlob->GetDesc().HasEqualDimensions( secondBlob->GetDesc() ) );

	IMathEngine& mathEngine = first->GetMathEngine();

	const CTapeBlob* tapeBlob1 = dynamic_cast<const CTapeBlob*>( firstBlob.Ptr() );
	IGradientTape* tape1 = tapeBlob1 != 0 ? tapeBlob1->Tape() : 0;
	const CTapeBlob* tapeBlob2 = dynamic_cast<const CTapeBlob*>( secondBlob.Ptr() );
	IGradientTape* tape2 = tapeBlob2 != 0 ? tapeBlob2->Tape() : 0;

	NeoAssert( tape1 == 0 || tape2 == 0 || tape1 == tape2 );

	IGradientTape* tape = tape1 != 0 ? tape1 : tape2;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, firstBlob->GetDesc() ) );
	mathEngine.VectorEltwiseDivide( firstBlob->GetData(), secondBlob->GetData(), result->GetData(), result->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeDiv( firstBlob, secondBlob ) ); 
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

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

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

CPtr<CDnnBlob> CTapeMax::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> jacobian = callJacobian( first, var );
	if( jacobian == 0 ) {
		return 0;
	}

	jacobian->GetMathEngine().VectorMaxDiff( first->GetData(), second, jacobian->GetData(),
		jacobian->GetObjectCount(), jacobian->GetObjectSize() );
	return jacobian;
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
	explicit CTapeSum( const CDnnBlob& first, const CArray<int>& axes );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

	static CPtr<CTapeBlob> Impl( const CDnnBlob* blob, const CArray<int>& axes, IGradientTape* tape );
	static CPtr<CDnnBlob> JacobianImpl( const CDnnBlob* blob, const CArray<int>& axes, const CTapeBlob* var );
private:
	CPtr<const CDnnBlob> first;
	CArray<int> axes;
};

CTapeSum::CTapeSum( const CDnnBlob& _first, const CArray<int>& _axes ) :
	first( &_first )
{
	_axes.CopyTo( axes );
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CTapeBlob> CTapeSum::Impl( const CDnnBlob* blob, const CArray<int>& axes, IGradientTape* tape )
{
	NeoAssert( blob != 0 );
	for( int i = 0; i < axes.Size(); i++ ) {
		NeoAssert( axes[i] >= -1 && axes[i] < BD_Count );
	}
	NeoAssert( isSequentialAxes( blob, axes ) );

	IMathEngine& mathEngine = blob->GetMathEngine();

	CPtr<CTapeBlob> result;
	if( axes.IsEmpty() ) {
		result = new CTapeBlob( tape, mathEngine, CBlobDesc( { 1 } ) );
		mathEngine.VectorSum( blob->GetData(), blob->GetDataSize(), result->GetData() );
	} else {
		int precedingDimension;
		int dimension;
		int followingDimension;
		getSequentialAxesDimensions( blob, axes, followingDimension, dimension, precedingDimension );
		CBlobDesc desc = blob->GetDesc();
		for( int i = 0; i < axes.Size(); i++ ) {
			desc.SetDimSize( axes[i], 1 );
		}
		result = new CTapeBlob( tape, mathEngine, desc );
		mathEngine.VectorSumAlongDimension( blob->GetData(), precedingDimension, dimension, followingDimension, result->GetData() );
	}

	return result;
}

CPtr<CDnnBlob> CTapeSum::JacobianImpl( const CDnnBlob* blob, const CArray<int>& axes, const CTapeBlob* var )
{
	NeoPresume( isSequentialAxes( blob, axes ) );

	CPtr<CDnnBlob> jacobian = callJacobian( blob, var );
	if( jacobian == 0 ) {
		return 0;
	}
	int height = jacobian->GetObjectCount();
	int width = jacobian->GetObjectSize();

	IMathEngine& mathEngine = jacobian->GetMathEngine();

	CPtr<CDnnBlob> result;
	if( axes.IsEmpty() ) {
		if( height == 1 ) {
			return jacobian;
		}
		result = CDnnBlob::CreateBlob( mathEngine, { width } );
		mathEngine.SumMatrixRows( 1, result->GetData(), jacobian->GetData(), height, width );
	} else {
		int precedingDimension;
		int dimension;
		int followingDimension;
		getSequentialAxesDimensions( blob, axes, followingDimension, dimension, precedingDimension );
		int resultHeight = followingDimension * precedingDimension;
		if( height == 1 && resultHeight == 1 ) {
			return jacobian;
		}
		result = CDnnBlob::CreateBlob( jacobian->GetMathEngine(), { resultHeight, 1, 1, 1, 1, 1, width } );
		if( height == 1 ) {
			mathEngine.VectorSumAlongDimensionDiag( jacobian->GetData(), precedingDimension, dimension,
				followingDimension, result->GetData() );
		} else {
			mathEngine.VectorSumAlongDimension( jacobian->GetData(), precedingDimension * width, dimension,
				followingDimension, result->GetData() );
		}
	}
	return result;
}

CPtr<CDnnBlob> CTapeSum::Jacobian( const CTapeBlob* var ) const
{
	return JacobianImpl( first, axes, var );
}

CPtr<const CDnnBlob> Sum( const CDnnBlob* first, const CArray<int>& _axes )
{
	CArray<int> axes;
	_axes.CopyTo( axes );
	axes.QuickSort<Ascending<int>>();

	if( !isSequentialAxes( first, axes ) ) {
		CPtr<const CDnnBlob> result = first;
		for( int i = 0; i < axes.Size(); i++ ) {
			result = Sum( result, { axes[i] } );
		}
		return result.Ptr();
	} else {
		const CTapeBlob* tapeBlob = dynamic_cast< const CTapeBlob* >( first );
		IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

		CPtr<CTapeBlob> result = CTapeSum::Impl( first, axes, tape );

		if( tape != 0 ) {
			CPtr<ITapeOperation> operation( new CTapeSum( *tapeBlob, axes ) );
			tape->Add( result, operation );
		}
		return result.Ptr();
	}
}

//------------------------------------------------------------------------------------------------------------

class CTapeCumSum : public ITapeOperation {
public:
	explicit CTapeCumSum( const CDnnBlob& first, int axis );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;
private:
	CPtr<const CDnnBlob> first;
	int axis;
};

CTapeCumSum::CTapeCumSum( const CDnnBlob& _first, int _axis ) :
	first( &_first ),
	axis( _axis )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeCumSum::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> jacobian = callJacobian( first, var );
	if( jacobian == 0 ) {
		return 0;
	}
	int height = jacobian->GetObjectCount();
	int width = jacobian->GetObjectSize();
	int precedingDimension;
	int dimension;
	int followingDimension;
	getSequentialAxesDimensions( first, { axis }, followingDimension, dimension, precedingDimension );

	if( height == 1 && dimension == first->GetDataSize() ) {
		return jacobian;
	}

	IMathEngine& mathEngine = jacobian->GetMathEngine();
	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( jacobian->GetMathEngine(), { first->GetDataSize(), 1, 1, 1, 1, 1, width } );
	if( height == 1 ) {
		mathEngine.VectorCumSumAlongDimensionDiag( jacobian->GetData(), precedingDimension, dimension,
			followingDimension, result->GetData() );
	} else {
		mathEngine.VectorCumSumAlongDimension( jacobian->GetData(), precedingDimension * width, dimension,
			followingDimension, result->GetData(), false );
	}
	return result;
}

CPtr<const CDnnBlob> CumSum( const CDnnBlob* first, int axis )
{
	NeoAssert( first != 0 );
	NeoAssert( axis >= 0 && axis < BD_Count );

	IMathEngine& mathEngine = first->GetMathEngine();

	const CTapeBlob* tapeBlob = dynamic_cast< const CTapeBlob* >( first );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

	int precedingDimension;
	int dimension;
	int followingDimension;
	getSequentialAxesDimensions( first, { axis }, followingDimension, dimension, precedingDimension );
	CPtr<CTapeBlob> result = new CTapeBlob( tape, mathEngine, first->GetDesc() );
	mathEngine.VectorCumSumAlongDimension( first->GetData(), precedingDimension, dimension, followingDimension,
		result->GetData(), false );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeCumSum( *tapeBlob, axis ) );
		tape->Add( result, operation );
	}

	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeMean : public ITapeOperation {
public:
	explicit CTapeMean( const CDnnBlob& first, const CArray<int>& axes );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

	static void DivideByCount( const CDnnBlob* in, CDnnBlob* out, const CArray<int>& axes );
private:
	CPtr<const CDnnBlob> first;
	CArray<int> axes;
};

CTapeMean::CTapeMean( const CDnnBlob& _first, const CArray<int>& _axes ) :
	first( &_first )
{
	_axes.CopyTo( axes );
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

void CTapeMean::DivideByCount( const CDnnBlob* first, CDnnBlob* result, const CArray<int>& axes )
{
	CPtr<CDnnBlob> div = CDnnBlob::CreateVector( result->GetMathEngine(), CT_Float, 1 );
	if( axes.IsEmpty() ) {
		div->GetData().SetValue( 1.f / static_cast< float >( first->GetDataSize() ) );
	} else {
		int count = 1;
		for( int i = 0; i < axes.Size(); i++ ) {
			count *= first->DimSize( axes[i] );
		}
		div->GetData().SetValue( 1.f / static_cast< float >( count ) );
	}
	result->GetMathEngine().VectorMultiply( result->GetData(), result->GetData(), result->GetDataSize(), div->GetData() );
}

CPtr<CDnnBlob> CTapeMean::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> jacobian = CTapeSum::JacobianImpl( first, axes, var );
	DivideByCount( first, jacobian, axes );
	return jacobian;
}

CPtr<const CDnnBlob> Mean( const CDnnBlob* first, const CArray<int>& _axes )
{
	CArray<int> axes;
	_axes.CopyTo( axes );
	axes.QuickSort<Ascending<int>>();

	if( !isSequentialAxes( first, axes ) ) {
		CPtr<const CDnnBlob> result = first;
		for( int i = 0; i < axes.Size(); i++ ) {
			result = Mean( result, { axes[i] } );
		}
		return result.Ptr();
	} else {
		const CTapeBlob* tapeBlob = dynamic_cast< const CTapeBlob* >( first );
		IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

		CPtr<CTapeBlob> result = CTapeSum::Impl( first, axes, tape );
		CTapeMean::DivideByCount( first, result, axes );

		if( tape != 0 ) {
			CPtr<ITapeOperation> operation( new CTapeMean( *tapeBlob, axes ) );
			tape->Add( result, operation );
		}
		return result.Ptr();
	}
}

//------------------------------------------------------------------------------------------------------------

class CTapeNeg : public ITapeOperation {
public:
	explicit CTapeNeg( const CDnnBlob& first );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
};

CTapeNeg::CTapeNeg( const CDnnBlob& _first ) :
	first( &_first )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeNeg::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> jacobian = callJacobian( first, var );
	if( jacobian == 0 ) {
		return 0;
	}

	jacobian->GetMathEngine().VectorNeg( jacobian->GetData(), jacobian->GetData(), jacobian->GetDataSize() );
	return jacobian;
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

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
};

CTapeAbs::CTapeAbs( const CDnnBlob& _first ) :
	first( &_first )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeAbs::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> jacobian = callJacobian( first, var );
	if( jacobian == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = first->GetMathEngine();
	mathEngine.VectorAbsDiff( jacobian->GetData(), jacobian->GetObjectCount(), jacobian->GetObjectSize(),
		first->GetData(), jacobian->GetData() );

	return jacobian;
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

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
};

CTapeExp::CTapeExp( const CDnnBlob& _first ) :
	first( &_first )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeExp::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> jacobian = callJacobian( first, var );
	if( jacobian == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = first->GetMathEngine();
	CFloatHandleStackVar expV( mathEngine, first->GetDataSize() );
	mathEngine.VectorExp( first->GetData(), expV, first->GetDataSize() );

	if( jacobian->GetObjectCount() == 1 ) {
		NeoAssert( jacobian->GetDataSize() == first->GetDataSize() );
		mathEngine.VectorEltwiseMultiply( jacobian->GetData(), expV, jacobian->GetData(), jacobian->GetDataSize() );
		return jacobian;
	}

	CPtr<CDnnBlob> result = jacobian->GetClone();
	mathEngine.MultiplyDiagMatrixByMatrix( expV, first->GetDataSize(), jacobian->GetData(), jacobian->GetObjectSize(),
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

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
};

CTapeLog::CTapeLog( const CDnnBlob& _first ) :
	first( &_first )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeLog::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> jacobian = callJacobian( first, var );
	if( jacobian == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = first->GetMathEngine();

	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, jacobian->GetDesc() );
	mathEngine.VectorLogDiff( jacobian->GetData(), jacobian->GetObjectCount(), jacobian->GetObjectSize(), first->GetData(), result->GetData() );
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

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

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

CPtr<CDnnBlob> CTapeTopK::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> jacobian = callJacobian( first, var );
	if( jacobian == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = first->GetMathEngine();

	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, {indices->GetDataSize(), 1, 1, 1, 1, 1, jacobian->GetObjectSize()} );
	mathEngine.VectorTopKDiff( jacobian->GetData(), jacobian->GetObjectCount(), jacobian->GetObjectSize(), indices->GetData<int>(),
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
	explicit CTapeClip( const CDnnBlob& first, float minValue, float maxValue );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

private:
	CPtr<const CDnnBlob> first;
	float minValue;
	float maxValue;
};

CTapeClip::CTapeClip( const CDnnBlob& _first, float _minValue, float _maxValue ) :
	first( &_first ),
	minValue( _minValue ),
	maxValue( _maxValue )
{
	NeoAssert( dynamic_cast<const CTapeBlob*>(first.Ptr()) != 0 );
}

CPtr<CDnnBlob> CTapeClip::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> jacobian = callJacobian( first, var );
	if( jacobian == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = first->GetMathEngine();

	CFloatHandleStackVar minHandle( mathEngine, 1 );
	minHandle.SetValue( minValue );
	CFloatHandleStackVar maxHandle( mathEngine, 1 );
	maxHandle.SetValue( maxValue );
	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, jacobian->GetDesc() );
	mathEngine.VectorMinMaxDiff( jacobian->GetData(), jacobian->GetObjectCount(), jacobian->GetObjectSize(), first->GetData(),
		result->GetData(), minHandle, maxHandle );
	return result.Ptr();
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
		CPtr<ITapeOperation> operation( new CTapeClip( *tapeBlob, minValue, maxValue ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeConcat : public ITapeOperation {
public:
	explicit CTapeConcat( const CObjectArray<CDnnBlob>& _blobs, int _axis );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

private:
	CObjectArray<CDnnBlob> blobs;
	int axis;
};

CTapeConcat::CTapeConcat( const CObjectArray<CDnnBlob>& _blobs, int _axis ) :
	axis( _axis )
{
	NeoAssert( axis >= 0 && axis < BD_Count );
	_blobs.CopyTo( blobs );
}

CPtr<CDnnBlob> CTapeConcat::Jacobian( const CTapeBlob* var ) const
{
	IMathEngine& mathEngine = blobs[0]->GetMathEngine();
	CObjectArray<CDnnBlob> jacobians;
	jacobians.SetSize( blobs.Size() );
	int width = 1;
	int resultAxis = 0;
	for( int i = 0; i < blobs.Size(); i++ ) {
		jacobians[i] = callJacobian( blobs[i], var );
		if( jacobians[i] != 0 ) {
			width = jacobians[i]->GetObjectSize();
		}
	}

	for( int i = 0; i < blobs.Size(); i++ ) {
		resultAxis += blobs[i]->DimSize( axis );
		if( jacobians[i] == 0 ) {
			CBlobDesc desc = blobs[i]->GetDesc();
			desc.SetDimSize( BD_Channels, desc.DimSize( BD_Channels ) * width );
			jacobians[i] = CDnnBlob::CreateBlob( mathEngine, desc );
			mathEngine.VectorFill( jacobians[i]->GetData(), 0.f, jacobians[i]->GetDataSize() );
		} else {
			if( jacobians[i]->GetObjectCount() == 1 ) {
				jacobians[i] = diagJacobianToFull( jacobians[i] );
			}
			CBlobDesc desc = blobs[i]->GetDesc();
			desc.SetDimSize( BD_Channels, desc.DimSize( BD_Channels ) * width );
			jacobians[i]->ReinterpretDimensions( desc );
		}
	}

	CBlobDesc desc = jacobians[0]->GetDesc();
	desc.SetDimSize( axis, resultAxis );
	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, desc );
	CDnnBlob::MergeByDim( mathEngine, static_cast<TBlobDim>( axis ), jacobians, result.Ptr() );
	result->ReinterpretDimensions( { result->GetDataSize() / width, 1, 1, 1, 1, 1, width } );
	return result.Ptr();
}

CPtr<const CDnnBlob> Concat( const CObjectArray<CDnnBlob>& blobs, int axis )
{
	IMathEngine& mathEngine = blobs[0]->GetMathEngine();
	const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( blobs[0].Ptr() );
	IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;
	CBlobDesc resultDesc = blobs[0]->GetDesc();
	int resultAxis = blobs[0]->DimSize( axis );
	for( int i = 1; i < blobs.Size(); i++ ) {
		const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( blobs[0].Ptr() );
		IGradientTape* tapeTemp = tapeBlob != 0 ? tapeBlob->Tape() : 0;
		if( tape == 0 ) {
			tape = tapeTemp;
		} else {
			NeoAssert( tapeTemp == 0 || tape == tapeTemp );
		}
		resultAxis += blobs[i]->DimSize( axis );
	}
	resultDesc.SetDimSize( axis, resultAxis );

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, resultDesc ) );
	CDnnBlob::MergeByDim( mathEngine, static_cast<TBlobDim>( axis ), blobs, result.Ptr() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapeConcat( blobs, axis ) ); 
		tape->Add( result, operation );
	}

	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapeBroadcast : public ITapeOperation {
public:
	explicit CTapeBroadcast( const CDnnBlob* first, const CBlobDesc& fromDesc );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;
private:
	CPtr<const CDnnBlob> first;
	CBlobDesc toDesc;
};

CTapeBroadcast::CTapeBroadcast( const CDnnBlob* _first, const CBlobDesc& _toDesc ) :
	first( _first ),
	toDesc( _toDesc )
{
}

CPtr<CDnnBlob> CTapeBroadcast::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> firstJacobian = callJacobian( first, var );
	if( firstJacobian == 0 ) {
		return 0;
	}

	IMathEngine& mathEngine = firstJacobian->GetMathEngine();
	const int height = firstJacobian->GetObjectCount();
	const int width = firstJacobian->GetObjectSize();

	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( first->GetMathEngine(), { toDesc.BlobSize(), 1, 1, 1, 1, 1, width } );
	if( height == 1 && first->GetDataSize() > 1 ) {
		firstJacobian = diagJacobianToFull( firstJacobian );
	}
	mathEngine.BroadcastCopy( result->GetData(), firstJacobian->GetData(),
		toDesc, first->GetDesc(), width );
	return result.Ptr();
}

CPtr<const CDnnBlob> Broadcast( const CDnnBlob* first, const CBlobDesc& toDesc )
{
	const CBlobDesc& firstDesc = first->GetDesc();
	int larger = 0;
	for( int i = 0; i < BD_Count; i++ ) {
		if( firstDesc.DimSize( i ) < toDesc.DimSize( i ) ) {
			NeoAssert( larger != 1 );
			larger = 2;
		} else if( firstDesc.DimSize( i ) > toDesc.DimSize( i ) ) {
			NeoAssert( larger != 2 );
			larger = 1;
		}
	}

	if( larger == 2 ) {
		const CTapeBlob* tapeBlob = dynamic_cast<const CTapeBlob*>( first );
		IGradientTape* tape = tapeBlob != 0 ? tapeBlob->Tape() : 0;

		IMathEngine& mathEngine = first->GetMathEngine();
		CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, toDesc ) );
		mathEngine.BroadcastCopy( result->GetData(), first->GetData(),
			toDesc, first->GetDesc(), 1 );

		if( tape != 0 ) {
			CPtr<ITapeOperation> operation( new CTapeBroadcast( first, toDesc ) );
			tape->Add( result, operation );
		}
		return result.Ptr();
	}

	return first;
}

//------------------------------------------------------------------------------------------------------------

void Reshape( CDnnBlob* first, const CBlobDesc& desc )
{
	NeoAssert( first != 0 );
	first->ReinterpretDimensions( desc );
}

//------------------------------------------------------------------------------------------------------------

CPtr<const CDnnBlob> Less( const CDnnBlob* first, const CDnnBlob* second )
{
	NeoAssert( first != 0 );
	NeoAssert( second != 0 );
	NeoAssert( first->GetDesc().HasEqualDimensions( second->GetDesc() ) );
	IMathEngine& mathEngine = first->GetMathEngine();

	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, first->GetDesc() );
	mathEngine.VectorEltwiseLess( first->GetData(), second->GetData(), result->GetData(), result->GetDataSize() );
	return result.Ptr();
}

CPtr<const CDnnBlob> Less( const CDnnBlob* first, float second )
{
	NeoAssert( first != 0 );
	IMathEngine& mathEngine = first->GetMathEngine();

	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, first->GetDesc() );
	mathEngine.VectorEltwiseLess( first->GetData(), second, result->GetData(), result->GetDataSize() );
	return result.Ptr();
}

CPtr<const CDnnBlob> Less( float first, const CDnnBlob* second )
{
	NeoAssert( second != 0 );
	IMathEngine& mathEngine = second->GetMathEngine();

	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, second->GetDesc() );
	mathEngine.VectorEltwiseLess( first, second->GetData(), result->GetData(), result->GetDataSize() );
	return result.Ptr();
}

//------------------------------------------------------------------------------------------------------------

class CTapePower : public ITapeOperation {
public:
	explicit CTapePower( const CDnnBlob* first, const CDnnBlob* second );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;
private:
	CPtr<const CDnnBlob> first;
	CPtr<const CDnnBlob> second;
};

CTapePower::CTapePower( const CDnnBlob* _first, const CDnnBlob* _second ) :
	first( _first ),
	second( _second )
{
}

CPtr<CDnnBlob> CTapePower::Jacobian( const CTapeBlob* var ) const
{
	CPtr<CDnnBlob> firstJacobian = callJacobian( first, var );
	CPtr<CDnnBlob> secondJacobian = callJacobian( second, var );

	IMathEngine& mathEngine = first->GetMathEngine();
	CPtr<CDnnBlob> temp = CDnnBlob::CreateBlob( first->GetMathEngine(), first->GetDesc() );

	if( firstJacobian == 0 && secondJacobian == 0 ) {
		return 0;
	}

	if( secondJacobian != 0 ) {
		mathEngine.VectorLog( first->GetData(), temp->GetData(), temp->GetDataSize() );
		mathEngine.VectorEltwiseMultiply( temp->GetData(), first->GetData(),
			temp->GetData(), temp->GetDataSize() );
		if( secondJacobian->GetObjectCount() == 1 ) {
			mathEngine.VectorEltwiseMultiply( temp->GetData(), secondJacobian->GetData(),
				secondJacobian->GetData(), secondJacobian->GetDataSize() );
		} else {
			mathEngine.MultiplyDiagMatrixByMatrix( temp->GetData(), temp->GetDataSize(),
				secondJacobian->GetData(), secondJacobian->GetObjectSize(), secondJacobian->GetData(),
				secondJacobian->GetDataSize() );
		}
	}

	if( firstJacobian != 0 ) {
		if( firstJacobian->GetObjectCount() == 1 ) {
			mathEngine.VectorEltwiseMultiply( second->GetData(), firstJacobian->GetData(),
				firstJacobian->GetData(), firstJacobian->GetDataSize() );
		} else {
			mathEngine.MultiplyDiagMatrixByMatrix( second->GetData(), second->GetDataSize(),
				firstJacobian->GetData(), firstJacobian->GetObjectSize(), firstJacobian->GetData(),
				firstJacobian->GetDataSize() );
		}
	}

	mathEngine.VectorSub( second->GetData(), 1.f, temp->GetData(), temp->GetDataSize() );
	mathEngine.VectorEltwisePower( first->GetData(), temp->GetData(), temp->GetData(), temp->GetDataSize() );

	if( firstJacobian == 0 ) {
		if( secondJacobian->GetObjectCount() == 1 ) {
			mathEngine.VectorEltwiseMultiply( temp->GetData(), secondJacobian->GetData(),
				secondJacobian->GetData(), secondJacobian->GetDataSize() );
			return secondJacobian;
		} else {
			mathEngine.MultiplyDiagMatrixByMatrix( temp->GetData(), temp->GetDataSize(),
				secondJacobian->GetData(), secondJacobian->GetObjectSize(), secondJacobian->GetData(),
				secondJacobian->GetDataSize() );
			return secondJacobian;
		}
	}
	if( secondJacobian == 0 ) {
		if( firstJacobian->GetObjectCount() == 1 ) {
			mathEngine.VectorEltwiseMultiply( temp->GetData(), firstJacobian->GetData(),
				firstJacobian->GetData(), firstJacobian->GetDataSize() );
			return firstJacobian;
		} else {
			mathEngine.MultiplyDiagMatrixByMatrix( temp->GetData(), temp->GetDataSize(),
				firstJacobian->GetData(), firstJacobian->GetObjectSize(), firstJacobian->GetData(),
				firstJacobian->GetDataSize() );
			return firstJacobian;
		}
	}

	if( firstJacobian->GetObjectCount() == 1 && secondJacobian->GetObjectCount() == 1 ) {
		mathEngine.VectorAdd( firstJacobian->GetData(), secondJacobian->GetData(), firstJacobian->GetData(), firstJacobian->GetDataSize() );
		mathEngine.VectorEltwiseMultiply( temp->GetData(), firstJacobian->GetData(),
			firstJacobian->GetData(), firstJacobian->GetDataSize() );
		return firstJacobian;
	} else if( firstJacobian->GetObjectCount() == 1 ) {
		mathEngine.AddDiagMatrixToMatrix( firstJacobian->GetData(), secondJacobian->GetData(),
			secondJacobian->GetObjectCount(), secondJacobian->GetObjectSize(), secondJacobian->GetData() );
		mathEngine.MultiplyDiagMatrixByMatrix( temp->GetData(), temp->GetDataSize(),
			secondJacobian->GetData(), secondJacobian->GetObjectSize(), secondJacobian->GetData(),
			secondJacobian->GetDataSize() );
		return secondJacobian;
	} else if( secondJacobian->GetObjectCount() == 1 ){
		mathEngine.AddDiagMatrixToMatrix( secondJacobian->GetData(), firstJacobian->GetData(),
			firstJacobian->GetObjectCount(), firstJacobian->GetObjectSize(), firstJacobian->GetData() );
		mathEngine.MultiplyDiagMatrixByMatrix( temp->GetData(), temp->GetDataSize(),
			firstJacobian->GetData(), firstJacobian->GetObjectSize(), firstJacobian->GetData(),
			firstJacobian->GetDataSize() );
		return firstJacobian;
	} else {
		mathEngine.VectorAdd( firstJacobian->GetData(), secondJacobian->GetData(), secondJacobian->GetData(),
			secondJacobian->GetDataSize() );
		mathEngine.MultiplyDiagMatrixByMatrix( temp->GetData(), temp->GetDataSize(),
			secondJacobian->GetData(), secondJacobian->GetObjectSize(), secondJacobian->GetData(),
			secondJacobian->GetDataSize() );
		return secondJacobian;
	}
}

CPtr<const CDnnBlob> Pow( const CDnnBlob* first, const CDnnBlob* second )
{
	NeoAssert( first != 0 );
	NeoAssert( second != 0 );

	IMathEngine& mathEngine = first->GetMathEngine();

	CBlobDesc broadcastedDesc = getBroadcastedDesc( first->GetDesc(), second->GetDesc() );
	CPtr<const CDnnBlob> firstBlob = Broadcast( first, broadcastedDesc );
	CPtr<const CDnnBlob> secondBlob = Broadcast( second, broadcastedDesc );

	NeoAssert( firstBlob->GetDesc().HasEqualDimensions( secondBlob->GetDesc() ) );

	const CTapeBlob* tapeBlob1 = dynamic_cast<const CTapeBlob*>( first );
	IGradientTape* tape1 = tapeBlob1 != 0 ? tapeBlob1->Tape() : 0;
	const CTapeBlob* tapeBlob2 = dynamic_cast<const CTapeBlob*>( second );
	IGradientTape* tape2 = tapeBlob2 != 0 ? tapeBlob2->Tape() : 0;

	NeoAssert( tape1 == 0 || tape2 == 0 || tape1 == tape2 );

	IGradientTape* tape = tape1 != 0 ? tape1 : tape2;

	CPtr<CTapeBlob> result( new CTapeBlob( tape, mathEngine, firstBlob->GetDesc() ) );
	mathEngine.VectorEltwisePower( firstBlob->GetData(), secondBlob->GetData(), result->GetData(), result->GetDataSize() );

	if( tape != 0 ) {
		CPtr<ITapeOperation> operation( new CTapePower( firstBlob, secondBlob ) );
		tape->Add( result, operation );
	}
	return result.Ptr();
}

CPtr<const CDnnBlob> Pow( const CDnnBlob* first, float _second )
{
	CPtr<CDnnBlob> second = CDnnBlob::CreateBlob( first->GetMathEngine(), { 1 } );
	second->GetData().SetValue( _second );
	return Pow( first, second );
}

CPtr<const CDnnBlob> Pow( float _first, const CDnnBlob* second )
{
	CPtr<CDnnBlob> first = CDnnBlob::CreateBlob( second->GetMathEngine(), { 1 } );
	first->GetData().SetValue( _first );
	return Pow( first, second );
}

//------------------------------------------------------------------------------------------------------------

CPtr<const CDnnBlob> BinaryCrossEntropy( const CDnnBlob* labels, const CDnnBlob* preds, bool fromLogits )
{
	NeoAssert( labels != 0 );
	NeoAssert( preds != 0 );
	NeoAssert( labels->GetDataSize() == preds->GetDataSize() );

	// Notations:
	// x = logits, z = labels

	// The original loss function formula:
	// loss = (1 - z) * x + log(1 + exp(-x))

	// The formula to avoid overflow for large exponent power in exp(-x):
	// loss = (1 - z) * x + log(1 + exp(-abs(x))) + max(-x, 0)

	CPtr<const CDnnBlob> clippedPreds = fromLogits ? preds : Clip( preds, 0.0000001f, 0.9999999f ).Ptr();

	CPtr<const CDnnBlob> x = fromLogits ? clippedPreds : Log( Div( clippedPreds, Sub(1, clippedPreds) ) );
	CPtr<const CDnnBlob> temp1 = Mul( Sub( 1, labels ), x );
	CPtr<const CDnnBlob> temp2 = Log( Add( 1, Exp( Neg( Abs(x) ) ) ) );
	CPtr<const CDnnBlob> temp3 = Max( Neg(x), 0 );

	return Add( Add( temp1, temp2 ), temp3 );
}

} // namespace NeoML
