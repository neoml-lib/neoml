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

namespace NeoML {

CTapeBlob::CTapeBlob( IGradientTape* _tape, const CDnnBlob& blob ) :
	CDnnBlob( blob.GetMathEngine(), blob.GetDesc(), blob.GetMathEngine().HeapAlloc( blob.GetDataSize() * sizeof(float) ), true ),
	tape( _tape )
{
	blob.GetMathEngine().VectorCopy( GetData(), blob.GetData(), blob.GetDataSize() );
}

CTapeBlob::CTapeBlob( IGradientTape* _tape, IMathEngine& mathEngine, const CBlobDesc& desc ) :
	CDnnBlob( mathEngine, desc, mathEngine.HeapAlloc( desc.BlobSize() * sizeof(float) ), true ),
	tape( _tape )
{
}

CTapeBlob::~CTapeBlob()
{
	Detach();
}

void CTapeBlob::Detach() const
{
	if( tape != 0 ) {
		tape->Remove( this );
		tape = 0;
	}
}

//------------------------------------------------------------------------------------------------------------

class CTapeVar : public ITapeOperation {
public:
	explicit CTapeVar( const CTapeBlob& var );

	CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const override;

private:
	const CTapeBlob* variable;
	const CPtr<CDnnBlob> jacobian;
};

CTapeVar::CTapeVar( const CTapeBlob& var ) :
	variable( &var ),
	jacobian( CDnnBlob::CreateBlob( var.GetMathEngine(), { var.GetDataSize() } ) )
{
	jacobian->Fill( 1.0f );
}

CPtr<CDnnBlob> CTapeVar::Jacobian( const CTapeBlob* var ) const
{
	if( variable == var ) {
		return jacobian->GetCopy();
	}
	return 0;
}

//------------------------------------------------------------------------------------------------------------

class CGradientTapeImpl : public IGradientTape {
public:

	void Add( const CTapeBlob* result, const ITapeOperation* operation ) override;
	void Remove( const CTapeBlob* result ) override;
	CPtr<const ITapeOperation> GetOperation( const CTapeBlob* expression ) override;

	void RemoveAllBlobs();

protected:
	virtual ~CGradientTapeImpl() { NeoPresume( operations.IsEmpty() ); }

private:
	CMap<const CTapeBlob*, CPtr<const ITapeOperation>> operations;
};

void CGradientTapeImpl::Add( const CTapeBlob* result, const ITapeOperation* operation )
{
	NeoAssert( result != 0 );
	NeoAssert( operation != 0 );

	CPtr<const ITapeOperation>& tapeOperation = operations.GetOrCreateValue( result );
	NeoAssert( tapeOperation == 0 );
	tapeOperation = operation;
}

void CGradientTapeImpl::Remove( const CTapeBlob* result )
{
	NeoAssert( result != 0 );
	operations.Delete( result );
}

void CGradientTapeImpl::RemoveAllBlobs()
{
	while( !operations.IsEmpty() ) {
		TMapPosition pos = operations.GetFirstPosition();
		CPtr<const CTapeBlob> ptr;
		if( ptr.PinWeakPtr( operations.GetKey( pos ) ) ) {
			ptr->Detach();
		}
	}
}

CPtr<const ITapeOperation> CGradientTapeImpl::GetOperation( const CTapeBlob* expression )
{
	NeoAssert( expression->Tape() == this );

	TMapPosition pos = operations.GetFirstPosition( expression );
	if( pos == NotFound ) {
		return 0;
	}

	return operations.GetValue( pos );
}

//------------------------------------------------------------------------------------------------------------

CGradientTape::CGradientTape() :
	impl( new CGradientTapeImpl() )
{
}

CGradientTape::~CGradientTape()
{
	NeoPresume( impl != 0 );
	impl->RemoveAllBlobs();
}

CPtr<const CDnnBlob> CGradientTape::Variable( const CDnnBlob& blob )
{
	NeoAssert( impl != 0 );
	CPtr<CTapeBlob> tapeBlob( new CTapeBlob( impl, blob ) );
	CPtr<CTapeVar> tapeOperation( new CTapeVar( *tapeBlob ) );
	impl->Add( tapeBlob, tapeOperation );
	return tapeBlob.Ptr();
}

CPtr<const CDnnBlob> CGradientTape::Gradient( const CDnnBlob& expression, const CDnnBlob& var )
{
	const CTapeBlob* expressionTapeBlob = dynamic_cast<const CTapeBlob*>( &expression );
	const CTapeBlob* varTapeBlob = dynamic_cast<const CTapeBlob*>( &var );

	if( expressionTapeBlob == 0 || varTapeBlob == 0 || expressionTapeBlob->Tape() == 0 || varTapeBlob->Tape() == 0 ) {
		return 0;
	}

	NeoAssert( expressionTapeBlob->Tape() == impl );
	NeoAssert( varTapeBlob->Tape() == impl );

	CPtr<const ITapeOperation> operation = impl->GetOperation( expressionTapeBlob );
	CPtr<const CDnnBlob> grad( operation->Jacobian( varTapeBlob ) );

	if( grad->GetObjectCount() == 1 ) {
		return grad;
	}

	CPtr<CDnnBlob> result( CDnnBlob::CreateBlob( grad->GetMathEngine(), var.GetDesc() ) );
	NeoAssert( var.GetDataSize() == grad->GetObjectSize() );
	grad->GetMathEngine().SumMatrixRows( 1, result->GetData(), grad->GetData(), grad->GetObjectCount(), grad->GetObjectSize() );
	return result.Ptr();
}

} // namespace NeoML
