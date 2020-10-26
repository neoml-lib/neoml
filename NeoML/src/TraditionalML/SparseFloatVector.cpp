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

#include <NeoML/TraditionalML/SparseFloatVector.h> 

namespace NeoML {

CSparseFloatVectorDesc CSparseFloatVectorDesc::Empty; // an empty vector descriptor

CSparseFloatVector::CSparseFloatVectorBody* CSparseFloatVector::CSparseFloatVectorBody::Duplicate() const
{
	CSparseFloatVectorBody* body = FINE_DEBUG_NEW CSparseFloatVectorBody( BufferSize );
	body->Desc.Size = Desc.Size;
	body->indexes.CopyFrom( Desc.Indexes, Desc.Size );
	body->values.CopyFrom( Desc.Values, Desc.Size );
	return body;
}

CSparseFloatVector::CSparseFloatVectorBody::CSparseFloatVectorBody( int bufferSize ) :
	BufferSize( bufferSize ),
	indexes( BufferSize ),
	values( BufferSize )
{
	Desc.Size = 0;
	Desc.Indexes = indexes;
	Desc.Values = values;
}

CSparseFloatVector::CSparseFloatVectorBody::CSparseFloatVectorBody( const CSparseFloatVectorDesc& desc ) :
	BufferSize( desc.Size ),
	indexes( BufferSize ),
	values( BufferSize )
{
	Desc.Size = desc.Size;
	indexes.CopyFrom( desc.Indexes, Desc.Size );
	values.CopyFrom( desc.Values, Desc.Size );
	Desc.Indexes = indexes;
	Desc.Values = values;
}

//------------------------------------------------------------------------------------------------------------

// Calculates the number of elements of two sparse vectors union
static inline int calcUnionElementsCount( const CSparseFloatVector& vector1, const CSparseFloatVector& vector2 )
{
	const CSparseFloatVectorDesc& body1 = vector1.GetDesc();
	const int size1 = vector1.NumberOfElements();
	const CSparseFloatVectorDesc& body2 = vector2.GetDesc();
	const int size2 = vector2.NumberOfElements();

	int i = 0;
	int j = 0;
	int count = 0; // the number of common elements
	while( i < size1 && j < size2 ) {
		if( body1.Indexes[i] == body2.Indexes[j] ) {
			i++;
			j++;
			count++;
		} else if( body1.Indexes[i] < body2.Indexes[j] ) {
			i++;
		} else {
			j++;
		}
	}
	return size1 + size2 - count;
}

const int sparseSignature = -1;
const int denseSignature = -2;
const int CSparseFloatVector::InitialBufferSize;

CSparseFloatVector::CSparseFloatVector()
{
	static_assert( InitialBufferSize >= 0, "InitialBufferSize < 0" );
}

CSparseFloatVector::CSparseFloatVector( int bufferSize ) :
	body( 0 )
{
	NeoAssert( bufferSize >= 0 );
	if( bufferSize > 0 ) {
		body = FINE_DEBUG_NEW CSparseFloatVectorBody( bufferSize );
	}
}

CSparseFloatVector::CSparseFloatVector( const CSparseFloatVectorDesc& desc ) :
	body( FINE_DEBUG_NEW CSparseFloatVectorBody( desc ) )
{
}

CSparseFloatVector::CSparseFloatVector( const CSparseFloatVector& vector ) :
	body( vector.body )
{
}

double CSparseFloatVector::Norm() const
{
	const int size = NumberOfElements();
	if( size == 0 ) {
		return 0;
	}
	const CSparseFloatVectorDesc& desc = GetDesc();
	
	double scale = 0.0;
	double ssq = 1.0;
	/* The following loop is equivalent to this call to the LAPACK 
		auxiliary routine:   CALL SLASSQ( N, X, INCX, SCALE, SSQ ) */
	for(int i = 0; i < size; i++) {
		if(desc.Values[i] != 0.0) {
			double abs = fabs(desc.Values[i]);
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

double CSparseFloatVector::NormL1() const
{
	const int size = NumberOfElements();
	const CSparseFloatVectorDesc& desc = GetDesc();
	double sum = 0;
	for( int i = 0; i < size; i++ ) {
		sum += fabs( desc.Values[i] );
	}
	return sum;
}

float CSparseFloatVector::MaxAbs() const
{
	float maxAbs = 0;
	const CSparseFloatVectorDesc& desc = GetDesc();
	const int size = NumberOfElements();
	for( int i = 0; i < size; i++ ) {
		maxAbs = max( maxAbs, static_cast<float>( abs( desc.Values[i] ) ) );
	}
	return maxAbs;
}

void CSparseFloatVector::SetAt( int index, float value )
{
	const int size = NumberOfElements();
	const CSparseFloatVectorDesc& desc = GetDesc();

	int i = NotFound;
	if( size == 0 || desc.Indexes[size - 1] <= index ) {
		i = size;
	} else {
		i = FindInsertionPoint<int, Ascending<int>, int>(index, desc.Indexes, size);
	}

	if( i > 0 && desc.Indexes[i - 1] == index ) {
		// this index already exists
		body.CopyOnWrite()->Desc.Values[i - 1] = value;
		return;
	}

	if( body != 0 && body->Desc.Size < body->BufferSize ) {
		CSparseFloatVectorBody* bodyPtr = body.CopyOnWrite();
		memmove(&bodyPtr->Desc.Indexes[i + 1], &bodyPtr->Desc.Indexes[i], (bodyPtr->Desc.Size - i) * sizeof(int));
		memmove(&bodyPtr->Desc.Values[i + 1], &bodyPtr->Desc.Values[i], (bodyPtr->Desc.Size - i) * sizeof(float));
		bodyPtr->Desc.Indexes[i] = index;
		bodyPtr->Desc.Values[i] = value;
		bodyPtr->Desc.Size += 1;
	} else {
		// Expand the array
		CSparseFloatVectorBody* newBody = FINE_DEBUG_NEW CSparseFloatVectorBody( max( (size * 3 + 1) / 2, InitialBufferSize) );

		memcpy( newBody->Desc.Indexes, desc.Indexes, i * sizeof(int) );
		memcpy( newBody->Desc.Values, desc.Values, i * sizeof(float) );

		newBody->Desc.Indexes[i] = index;
		newBody->Desc.Values[i] = value;
		memcpy( &newBody->Desc.Indexes[i + 1], &desc.Indexes[i], (size - i) * sizeof(int) );
		memcpy( &newBody->Desc.Values[i + 1], &desc.Values[i], (size - i) * sizeof(float) );

		newBody->Desc.Size = size + 1;
		body = newBody;
	}
}

bool CSparseFloatVector::GetValue( int index, float& value ) const
{
	const int size = NumberOfElements();
	const CSparseFloatVectorDesc& desc = GetDesc();

	const int pos = FindInsertionPoint<int, Ascending<int>, int>( index, desc.Indexes, size ) - 1;
	if( pos >= 0 && desc.Indexes[pos] == index ) {
		value = desc.Values[pos];
		return true;
	}
	value = 0;
	return false;
}

float CSparseFloatVector::GetValue( int index ) const
{
	float result = 0;
	if( GetValue( index, result ) ) {
		return result;
	}
	return 0;
}

void CSparseFloatVector::Nullify()
{
	if( body != 0 ) {
		body.CopyOnWrite()->Desc.Size = 0;
	}
}

CSparseFloatVector& CSparseFloatVector::operator = ( const CSparseFloatVector& vector )
{
	body = vector.body;
	return *this;
}

CSparseFloatVector& CSparseFloatVector::operator += ( const CSparseFloatVector& vector )
{
	const int otherSize = vector.NumberOfElements();
	if( otherSize == 0 ) {
		return *this;
	}
	const CSparseFloatVectorDesc& otherDesc = vector.GetDesc();

	const int size = NumberOfElements();
	if( size == 0 ) {
		*this = vector;
		return *this;
	}
	const CSparseFloatVectorDesc& desc = GetDesc();

	const int newSize = calcUnionElementsCount( *this, vector );
	CSparseFloatVectorBody* newBody = FINE_DEBUG_NEW CSparseFloatVectorBody( newSize );

	int i = 0;
	int j = 0;
	int k = 0;
	while( i < size && j < otherSize ) {
		if( desc.Indexes[i] == otherDesc.Indexes[j] ) {
			newBody->Desc.Indexes[k] = desc.Indexes[i];
			newBody->Desc.Values[k] = desc.Values[i] + otherDesc.Values[j];
			i++;
			j++;
		} else if( desc.Indexes[i] < otherDesc.Indexes[j] ) {
			newBody->Desc.Indexes[k] = desc.Indexes[i];
			newBody->Desc.Values[k] = desc.Values[i];
			i++;
		} else {
			newBody->Desc.Indexes[k] = otherDesc.Indexes[j];
			newBody->Desc.Values[k] = otherDesc.Values[j];
			j++;
		}
		k++;
	}

	while( i < size ) {
		newBody->Desc.Indexes[k] = desc.Indexes[i];
		newBody->Desc.Values[k] = desc.Values[i];
		i++;
		k++;
	}

	while( j < otherSize ) {
		newBody->Desc.Indexes[k] = otherDesc.Indexes[j];
		newBody->Desc.Values[k] = otherDesc.Values[j];
		j++;
		k++;
	}

	newBody->Desc.Size = k;
	body = newBody;
	return *this;
}

CSparseFloatVector& CSparseFloatVector::operator -= ( const CSparseFloatVector& vector )
{
	const int otherSize = vector.NumberOfElements();
	if( otherSize == 0 ) {
		return *this;
	}
	const CSparseFloatVectorDesc& otherDesc = vector.GetDesc();

	const int size = NumberOfElements();
	const CSparseFloatVectorDesc& desc = GetDesc();

	const int elementsCount = calcUnionElementsCount( *this, vector );
	CSparseFloatVectorBody* newBody = FINE_DEBUG_NEW CSparseFloatVectorBody( elementsCount );

	int i = 0;
	int j = 0;
	int k = 0;
	while( i < size && j < otherSize ) {
		if( desc.Indexes[i] == otherDesc.Indexes[j] ) {
			newBody->Desc.Indexes[k] = desc.Indexes[i];
			newBody->Desc.Values[k] = desc.Values[i] - otherDesc.Values[j];
			i++;
			j++;
		} else if( desc.Indexes[i] < otherDesc.Indexes[j] ) {
			newBody->Desc.Indexes[k] = desc.Indexes[i];
			newBody->Desc.Values[k] = desc.Values[i];
			i++;
		} else {
			newBody->Desc.Indexes[k] = otherDesc.Indexes[j];
			newBody->Desc.Values[k] = -otherDesc.Values[j];
			j++;
		}
		k++;
	}

	while( i < size ) {
		newBody->Desc.Indexes[k] = desc.Indexes[i];
		newBody->Desc.Values[k] = desc.Values[i];
		i++;
		k++;
	}

	while( j < otherSize ) {
		newBody->Desc.Indexes[k] = otherDesc.Indexes[j];
		newBody->Desc.Values[k] = -otherDesc.Values[j];
		j++;
		k++;
	}

	newBody->Desc.Size = k;
	body = newBody;
	return *this;
}

CSparseFloatVector& CSparseFloatVector::operator *= ( double factor )
{
	CSparseFloatVectorDesc* desc = CopyOnWrite();
	const int size = NumberOfElements();
	for( int i = 0; i < size; i++ ) {
		desc->Values[i] = static_cast<float>( desc->Values[i] * factor );
	}
	return *this;
}

CSparseFloatVector& CSparseFloatVector::MultiplyAndAdd( const CSparseFloatVector& vector, double factor )
{
	const int otherSize = vector.NumberOfElements();
	if( otherSize == 0 ) {
		return *this;
	}
	const CSparseFloatVectorDesc& otherDesc = vector.GetDesc();

	const int size = NumberOfElements();
	const CSparseFloatVectorDesc& desc = GetDesc();

	const int newSize = calcUnionElementsCount( *this, vector );
	CSparseFloatVectorBody* newBody = FINE_DEBUG_NEW CSparseFloatVectorBody( newSize );

	int i = 0;
	int j = 0;
	int k = 0;
	while( i < size && j < otherSize ) {
		if( desc.Indexes[i] == otherDesc.Indexes[j] ) {
			newBody->Desc.Indexes[k] = desc.Indexes[i];
			newBody->Desc.Values[k] = static_cast<float>( desc.Values[i] + factor * desc.Values[j] );
			i++;
			j++;
		} else if( desc.Indexes[i] < otherDesc.Indexes[j] ) {
			newBody->Desc.Indexes[k] = desc.Indexes[i];
			newBody->Desc.Values[k] = desc.Values[i];
			i++;
		} else {
			newBody->Desc.Indexes[k] = otherDesc.Indexes[j];
			newBody->Desc.Values[k] = static_cast<float>( factor * otherDesc.Values[j] );
			j++;
		}
		k++;
	}

	while( i < size ) {
		newBody->Desc.Indexes[k] = desc.Indexes[i];
		newBody->Desc.Values[k] = desc.Values[i];
		i++;
		k++;
	}

	while( j < otherSize ) {
		newBody->Desc.Indexes[k] = otherDesc.Indexes[j];
		newBody->Desc.Values[k] = static_cast<float>( factor * otherDesc.Values[j] );
		j++;
		k++;
	}

	newBody->Desc.Size = k;
	body = newBody;
	return *this;
}

void CSparseFloatVector::SquareEachElement()
{
	CSparseFloatVectorDesc* desc = CopyOnWrite();
	const int size = NumberOfElements();
	for( int i = 0; i < size; i++ ) {
		desc->Values[i] *= desc->Values[i];
	}
}

void CSparseFloatVector::MultiplyBy( const CSparseFloatVector& factor )
{
	const int otherSize = factor.NumberOfElements();
	if( otherSize == 0 ) {
		return;
	}
	const CSparseFloatVectorDesc& otherDesc = factor.GetDesc();

	CSparseFloatVectorDesc* desc = CopyOnWrite();
	const int size = NumberOfElements();

	int i = 0;
	int j = 0;
	while( i < size && j < otherSize ) {
		if( desc->Indexes[i] == otherDesc.Indexes[j] ) {
			desc->Values[i] *= otherDesc.Values[j];
			i++;
			j++;
		} else if( desc->Indexes[i] < otherDesc.Indexes[j] ) {
			i++;
		} else {
			j++;
		}
	}
}

void CSparseFloatVector::DivideBy( const CSparseFloatVector& divisor )
{
	const int otherSize = divisor.NumberOfElements();
	if( otherSize == 0 ) {
		return;
	}
	const CSparseFloatVectorDesc& otherDesc = divisor.GetDesc();
	
	CSparseFloatVectorDesc* desc = CopyOnWrite();
	const int size = NumberOfElements();

	int i = 0;
	int j = 0;
	while( i < size && j < otherSize ) {
		if( desc->Indexes[i] == otherDesc.Indexes[j] ) {
			NeoPresume( otherDesc.Values[j] != 0 );
			desc->Values[i] /= otherDesc.Values[j];
			i++;
			j++;
		} else if( desc->Values[i] < otherDesc.Indexes[j] ) {
			i++;
		} else {
			j++;
		}
	}
}

void CSparseFloatVector::Serialize( CArchive& archive )
{
	if( archive.IsLoading() ) {
		int sign = archive.ReadSmallValue();
		check( sign == denseSignature || sign == sparseSignature, ERR_BAD_ARCHIVE, archive.Name() );

		if( sign == sparseSignature ) {
			int size = 0;
			archive >> size;

			if( size == 0 ) {
				body = 0;
				return;
			}
			if( size < 0 ) {
				check( false, ERR_BAD_ARCHIVE, archive.Name() );
			}

			CSparseFloatVectorBody* newBody = FINE_DEBUG_NEW CSparseFloatVectorBody( size );
			int elementIndex = 0;
			for( int i = 0; i < size; i++ ) {
				archive >> newBody->Desc.Indexes[elementIndex];
				float value = 0;
				archive >> value;
				if( value != 0.f ) {
					newBody->Desc.Values[elementIndex] = value;
					elementIndex++;
				}
			}
			newBody->Desc.Size = elementIndex;
			body = newBody;
		} else {
			int length = 0;
			archive >> length;
			int bodySize = 0;
			archive >> bodySize;
			CSparseFloatVectorBody* newBody = FINE_DEBUG_NEW CSparseFloatVectorBody( bodySize );
			int elementIndex = 0;
			for( int i = 0; i < length; ++i ) {
				float value;
				archive >> value;
				if( value != 0.f ) {
					newBody->Desc.Indexes[elementIndex] = i;
					newBody->Desc.Values[elementIndex] = value;
					elementIndex += 1;
				}
			}
			newBody->Desc.Size = elementIndex;
			body = newBody;
		}
	} else if( archive.IsStoring() ) {
		const CSparseFloatVectorDesc& desc = GetDesc();
		int notNullElementCount = 0;
		int lastNotNullElementIndex = NotFound;
		for( int i = 0; i < desc.Size; i++ ) {
			if( desc.Values[i] != 0.f ) {
				notNullElementCount++;
				lastNotNullElementIndex = i;
			}
		}

		// Expected serialization size
		const int denseSize = 2 * sizeof( int ) // the vector length and buffer size
			+ ( sizeof( float ) * ( notNullElementCount == 0 ? 0 : desc.Indexes[lastNotNullElementIndex] + 1 ) );
		const int sparseSize = sizeof( int ) // the buffer size
			+ ( ( sizeof( float ) + sizeof( int ) ) * notNullElementCount );

		if( sparseSize <= denseSize ) {
			archive.WriteSmallValue( sparseSignature );
			archive << notNullElementCount;
			for( int i = 0; i < desc.Size; i++ ) {
				if( desc.Values[i] != 0.f ) {
					archive << desc.Indexes[i];
					archive << desc.Values[i];
				}
			}
		} else {
			const int length = notNullElementCount == 0 ? 0 : desc.Indexes[lastNotNullElementIndex] + 1;
			archive.WriteSmallValue( denseSignature );
			archive << length;
			archive << notNullElementCount;
			for( int i = 0; i < length; ++i ) {
				archive << GetValue( i );
			}
		}
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
