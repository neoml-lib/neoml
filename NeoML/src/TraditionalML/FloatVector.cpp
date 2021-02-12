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

#include <NeoML/TraditionalML/FloatVector.h> 

namespace NeoML {

CFloatVector::CFloatVectorBody::CFloatVectorBody( int size )
{
	Values.SetSize( size );
	Desc.Values = Values.GetPtr();
	Desc.Size = size;
}

CFloatVector::CFloatVector( int size, const CSparseFloatVector& sparseVector )
{
	NeoAssert( size >= 0 );
	auto bodyPtr = FINE_DEBUG_NEW CFloatVectorBody( size );
	const CSparseFloatVectorDesc& desc = sparseVector.GetDesc();
	NeoAssert( desc.Indexes != nullptr );
	int ptrPos = 0;
	int ptrSize = sparseVector.NumberOfElements();

	float value = 0;
	for( int i = 0; i < size; i++ ) {
		if( ptrPos >= ptrSize || i < desc.Indexes[ptrPos] ) {
			value = 0;
		} else {
			value = desc.Values[ptrPos];
			ptrPos++;
		}
		bodyPtr->Desc.Values[i] = value;
	}

	// No elements should stay unprocessed!
	NeoAssert( ptrPos == ptrSize );

	body = bodyPtr;
}

CFloatVector::CFloatVector( int size, const CSparseFloatVectorDesc& desc )
{
	NeoAssert( size >= 0 );
	auto bodyPtr = FINE_DEBUG_NEW CFloatVectorBody( size );

	if( desc.Indexes == nullptr ) {
		NeoAssert( size == desc.Size );
		::memcpy( bodyPtr->Desc.Values, desc.Values, size * sizeof( float ) );
	} else {
		int ptrSize = desc.Size;
		int ptrPos = 0;
		float value = 0;
		for( int i = 0; i < size; i++ ) {
			if( ptrPos >= ptrSize || i < desc.Indexes[ptrPos] ) {
				value = 0;
			} else {
				value = desc.Values[ptrPos];
				ptrPos++;
			}
			bodyPtr->Desc.Values[i] = value;
		}
		// No elements should stay unprocessed!
		NeoAssert( ptrPos == ptrSize );
	}

	body = bodyPtr;
}

CFloatVector::CFloatVector( int size )
{
	NeoAssert( size >= 0 );
	body = FINE_DEBUG_NEW CFloatVectorBody( size );
}

CFloatVector::CFloatVector( int size, float init )
{
	NeoAssert( size >= 0 );

	auto bodyPtr = FINE_DEBUG_NEW CFloatVectorBody( size );
	for( int i = 0; i < size; ++i ) {
		bodyPtr->Desc.Values[i] = init;
	}

	body = bodyPtr;
}

CFloatVector::CFloatVector( const CFloatVector& vector )
{
	body = vector.body;
}

double CFloatVector::Norm() const
{
	const float* array = body->Desc.Values;
	const int size = body->Desc.Size;
    double scale = 0.0;
    double ssq = 1.0;
    /* The following loop is equivalent to this call to the LAPACK 
        auxiliary routine:   CALL SLASSQ( N, X, INCX, SCALE, SSQ ) */
	for(int i = 0; i < size; i++) {
		if(array[i] != 0.0) {
			double abs = fabs(array[i]);
			if(scale < abs) {
				double temp = scale / abs;
				ssq = ssq * (temp * temp) + 1.0;
				scale = abs;
			} else {
				double temp = abs / scale;
				ssq += temp * temp;
			}
		}
	}
	return scale * sqrt(ssq);
}

double CFloatVector::NormL1() const
{
	double sum = 0;
	const float* array = body->Desc.Values;
	const int size = body->Desc.Size;
	for( int i = 0; i < size; i++ ) {
		sum += fabs( array[i] );
	}
	return sum;
}

float CFloatVector::MaxAbs() const
{
	float maxAbs = 0;
	const float* array = body->Desc.Values;
	const int size = body->Desc.Size;
	for( int i = 0; i < size; i++ ) {
		maxAbs = max( maxAbs, static_cast<float>( abs( array[i] ) ) );
	}
	return maxAbs;
}

void CFloatVector::Nullify()
{
	NeoPresume( body->Desc.Size >= 0 );
	memset( CopyOnWrite(), 0, body->Desc.Size * sizeof( float ) );
}

CFloatVector& CFloatVector::operator = ( const CFloatVector& vector )
{
	body = vector.body;
	return *this;
}

CFloatVector& CFloatVector::operator += ( const CFloatVector& vector )
{
	NeoPresume( body->Desc.Size == vector.body->Desc.Size );
	NeoPresume( body->Desc.Size >= 0 );

	float* array = CopyOnWrite();
	const float* operand = vector.body->Desc.Values;
	const int size = body->Desc.Size;

	for( int i = 0; i < size; i++ ) {
		array[i] += operand[i];
	}
	return *this;
}

CFloatVector& CFloatVector::operator -= ( const CFloatVector& vector )
{
	NeoPresume( body->Desc.Size == vector.body->Desc.Size );
	NeoPresume( body->Desc.Size >= 0 );

	float* array = CopyOnWrite();
	const float* operand = vector.body->Desc.Values;
	const int size = body->Desc.Size;

	for( int i = 0; i < size; i++ ) {
		array[i] -= operand[i];
	}
	return *this;
}

CFloatVector& CFloatVector::operator *= ( double factor )
{
	NeoPresume( body->Desc.Size >= 0 );
	float* ptr = CopyOnWrite();
	const int size = body->Desc.Size;
	for( int i = 0; i < size; i++ ) {
		ptr[i] = static_cast<float>( ptr[i] * factor );
	}
	return *this;
}

void CFloatVector::SquareEachElement()
{	
	const int size = Size();
	float* ptr = CopyOnWrite();
	for( int i = 0; i < size; i++ ) {
		ptr[i] *= ptr[i];
	}
}

void CFloatVector::MultiplyBy( const CFloatVector& arg )
{	
	NeoPresume( arg.Size() == Size() );
	const int size = Size();
	
	const float* factor = arg.GetPtr();
	float* ptr = CopyOnWrite();
	for( int i = 0; i < size; i++ ) {
		ptr[i] *= factor[i];
	}
}

void CFloatVector::DivideBy( const CFloatVector& arg )
{
	NeoPresume( arg.Size() == Size() );
	const int size = Size();
	
	const float* divisor = arg.GetPtr();
	float* ptr = CopyOnWrite();
	for( int i = 0; i < size; i++ ) {
		NeoPresume( divisor[i] != 0 );
		ptr[i] /= divisor[i];
	}
}

void CFloatVector::Serialize( CArchive& archive )
{
	if( archive.IsLoading() ) {
		archive >> *this;
	} else {
		archive << *this;
	}
}

CFloatVector& CFloatVector::operator = ( const CSparseFloatVector& vector )
{
	float* ptr = CopyOnWrite();
	const CSparseFloatVectorDesc& desc = vector.GetDesc();
	NeoAssert( desc.Indexes != nullptr );
	const int size = body->Desc.Size;
	memset( ptr, 0, size * sizeof( float ) );
	const int numberOfElements = vector.NumberOfElements();
	for(int i = 0; i < numberOfElements; i++) {
		int j = desc.Indexes[i];
		if( j < size) {
			ptr[j] = desc.Values[i];
		}
	}
	return *this;
}

CFloatVector& CFloatVector::operator += ( const CSparseFloatVector& vector )
{
	float* ptr = CopyOnWrite();
	const CSparseFloatVectorDesc& desc = vector.GetDesc();
	NeoAssert( desc.Indexes != nullptr );
	const int size = body->Desc.Size;
	const int numberOfElements = vector.NumberOfElements();
	for(int i = 0; i < numberOfElements; i++) {
		int j = desc.Indexes[i];
		if( j < size) {
			ptr[j] += desc.Values[i];
		}
	}
	return *this;
}

CFloatVector& CFloatVector::operator -= ( const CSparseFloatVector& vector )
{
	float* ptr = CopyOnWrite();
	const CSparseFloatVectorDesc& desc = vector.GetDesc();
	NeoAssert( desc.Indexes != nullptr );
	const int size = body->Desc.Size;
	const int numberOfElements = vector.NumberOfElements();
	for(int i = 0; i < numberOfElements; i++) {
		int j = desc.Indexes[i];
		if( j < size) {
			ptr[j] -= desc.Values[i];
		}
	}
	return *this;
}

CFloatVector& CFloatVector::MultiplyAndAdd( const CSparseFloatVectorDesc& desc, double factor )
{
	float* ptr = CopyOnWrite();
	if( desc.Indexes != nullptr ) {
		const int size = body->Desc.Size;
		const int numberOfElements = desc.Size;
		for( int i = 0; i < numberOfElements; i++ ) {
			int j = desc.Indexes[i];
			if( j < size ) {
				ptr[j] = static_cast< float >( ptr[j] + desc.Values[i] * factor );
			}
		}
	} else { // dense inside
		NeoPresume( desc.Size <= body->Desc.Size );
		for( int i = 0; i < desc.Size; i++ ) {
			ptr[i] = static_cast< float >( ptr[i] + factor * desc.Values[i] );
		}
	}
	return *this;
}

CSparseFloatVector CFloatVector::SparseVector() const
{
	const float* ptr = GetPtr();
	const int size = Size();
	// Calculate how many non-zero elements there are
	int nonZero = 0;
	for( int i = 0; i < size; i++ ) {
		if( ptr[i] != 0 ) {
			nonZero += 1;
		}
	}
	CSparseFloatVector result(nonZero);
	for( int i = 0; i < size; i++ ) {
		if( ptr[i] != 0 ) {
			result.SetAt(i, ptr[i]);
		}
	}
	return result;
}

} // namespace NeoML
