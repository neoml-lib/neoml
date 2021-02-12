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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/SparseFloatVector.h>
#include <NeoML/TraditionalML/VectorIterator.h>

#include <cstddef>

namespace NeoML {

class CFloatVector;
typedef CArray<CFloatVector> CFloatVectorArray;

//------------------------------------------------------------------------------------------------------------

// Feature vector
class NEOML_API CFloatVector {
public:
	typedef CVectorIterator<const float> TConstIterator;
	typedef CVectorIterator<float> TIterator;

	CFloatVector() {}
	// Creates a vector of size length from the given sparse vector; 
	// the features that are not present in sparse vector are set to 0
	CFloatVector( int size, const CSparseFloatVector& sparseVector );
	CFloatVector( int size, const CSparseFloatVectorDesc& sparseVector );
	explicit CFloatVector( int size );
	CFloatVector( int size, float init );
	CFloatVector( const CFloatVector& other );

	bool IsNull() const { return body == nullptr; }
	int Size() const { return body->Values.Size(); }

	double Norm() const;
	double NormL1() const;
	float MaxAbs() const;

	float operator [] ( int i ) const { NeoPresume( i >= 0 && i < body->Values.Size() ); return body->Values[i]; }
	void SetAt( int i, float what );
	float* CopyOnWrite() { return body.CopyOnWrite()->Values.GetPtr(); }
	const float* GetPtr() const { return body->Values.GetPtr(); }
	const CSparseFloatVectorDesc& GetDesc() const
		{ return body == nullptr ? CSparseFloatVectorDesc::Empty : body->Desc; }

	void Nullify();
	
	CFloatVector& operator = ( const CFloatVector& vector );
	CFloatVector& operator += ( const CFloatVector& vector );
	CFloatVector& operator -= ( const CFloatVector& vector );
	CFloatVector& operator *= ( double factor );
	CFloatVector& operator /= ( double factor ) { return *this *= (1 / factor); }

	// Adds the given vector multiplied by factor
	CFloatVector& MultiplyAndAdd( const CFloatVector& vector, double factor );
	CFloatVector& MultiplyAndAdd( const CSparseFloatVector& vector, double factor ) { return MultiplyAndAdd( vector.GetDesc(), factor ); }
	CFloatVector& MultiplyAndAdd( const CSparseFloatVectorDesc& vector, double factor );

	// Adds the given vector, extended by one with the help of LinearFunction gradient, and then multiplied by factor
	CFloatVector& MultiplyAndAddExt( const CFloatVector& vector, double factor );
	CFloatVector& MultiplyAndAddExt( const CSparseFloatVector& vector, double factor );
	CFloatVector& MultiplyAndAddExt( const CSparseFloatVectorDesc& vector, double factor );

	// Elementwise operations:
	void SquareEachElement();
	void MultiplyBy( const CFloatVector& factor );
	void DivideBy( const CFloatVector& divisor );
	void Serialize( CArchive& archive );

	// Sparse vector operations:
	CFloatVector& operator = ( const CSparseFloatVector& vector );
	CFloatVector& operator += ( const CSparseFloatVector& vector );
	CFloatVector& operator -= ( const CSparseFloatVector& vector );
	// Gets the sparse vector built from this one
	// The zero feature values are not included into the sparse vector
	CSparseFloatVector SparseVector() const;

	friend CArchive& operator << ( CArchive& archive, const CFloatVector& vector );
	friend CArchive& operator >> ( CArchive& archive, CFloatVector& vector );

	TConstIterator begin() const;
	TConstIterator end() const;
	TIterator begin();
	TIterator end();

private:
	// The body of the vector is an object containing all its data.
	struct NEOML_API CFloatVectorBody: public IObject {
		CSparseFloatVectorDesc Desc;
		CFastArray<float, 1> Values;

		explicit CFloatVectorBody( int size );
	
		CFloatVectorBody* Duplicate() const;
	};

	CCopyOnWritePtr<CFloatVectorBody> body; // the vector body
	CSparseFloatVectorDesc desc;
};

inline CFloatVector::CFloatVectorBody* CFloatVector::CFloatVectorBody::Duplicate() const
{
	auto result = FINE_DEBUG_NEW CFloatVectorBody( Values.Size() );
	Values.CopyTo( result->Values );
	result->Desc.Values = result->Values.GetPtr();
	return result;
}

inline CFloatVector& CFloatVector::MultiplyAndAdd( const CFloatVector& vector, double factor )
{
	NeoPresume( body->Desc.Size == vector.body->Desc.Size );
	NeoPresume( body->Desc.Size >= 0 );

	MultiplyAndAdd( vector.GetDesc(), factor );
	return *this;
}

inline CFloatVector& CFloatVector::MultiplyAndAddExt( const CFloatVector& vector, double factor )
{
	NeoPresume( body->Desc.Size == vector.body->Desc.Size + 1 );
	NeoPresume( body->Desc.Size >= 0 );

	MultiplyAndAdd( vector.GetDesc(), factor );
	SetAt( Size() - 1, static_cast<float>( GetPtr()[Size() - 1] + factor ) );

	return *this;
}

inline CFloatVector& CFloatVector::MultiplyAndAddExt( const CSparseFloatVector& vector, double factor )
{
	NeoPresume( body->Desc.Size > vector.GetDesc().Indexes[vector.NumberOfElements()-1] + 1 );
	NeoPresume( body->Desc.Size >= 0 );

	MultiplyAndAdd( vector.GetDesc(), factor );
	SetAt( Size() - 1, static_cast<float>( GetPtr()[Size() - 1] + factor ) );

	return *this;
}

inline CFloatVector& CFloatVector::MultiplyAndAddExt( const CSparseFloatVectorDesc& vector, double factor )
{
	MultiplyAndAdd( vector, factor );
	SetAt( Size() - 1, static_cast<float>( GetPtr()[Size() - 1] + factor ) );

	return *this;
}

inline CFloatVector::TConstIterator CFloatVector::begin() const
{
	if( body == nullptr ) {
		return TConstIterator();
	}
	return TConstIterator( GetPtr() );
}

inline CFloatVector::TConstIterator CFloatVector::end() const
{
	if( body == nullptr ) {
		return TConstIterator();
	}
	return TConstIterator( GetPtr() + Size() );
}

inline CFloatVector::TIterator CFloatVector::begin()
{
	if( body == nullptr ) {
		return TIterator();
	}
	return TIterator( CopyOnWrite() );
}

inline CFloatVector::TIterator CFloatVector::end()
{
	if( body == nullptr ) {
		return TIterator();
	}
	return TIterator( CopyOnWrite() + Size() );
}

// The dot product of two vectors
inline double DotProduct( const CSparseFloatVectorDesc& vector1, const CSparseFloatVectorDesc& vector2 )
{
	double sum = 0;

	if( vector1.Indexes == nullptr ) {
		if( vector2.Indexes == nullptr ) {
			const int size = min( vector1.Size, vector2.Size );
			for( int i = 0; i < size; i++ ) {
				sum += static_cast<double>( vector1.Values[i] ) * vector2.Values[i];
			}
		} else {
			// The sparse vector may not be longer than the regular one
			NeoPresume( vector2.Size == 0 || vector2.Indexes[vector2.Size - 1] < vector1.Size );
			for( int i = 0; i < vector2.Size; i++ ) {
				sum += static_cast<double>( vector2.Values[i] ) * vector1.Values[vector2.Indexes[i]];
			}
		}
	} else {
		if( vector2.Indexes == nullptr ) {
			// The sparse vector may not be longer than the regular one
			NeoPresume( vector1.Size == 0 || vector1.Indexes[vector1.Size - 1] < vector2.Size );
			for( int i = 0; i < vector1.Size; i++ ) {
				sum += static_cast<double>( vector1.Values[i] ) * vector2.Values[vector1.Indexes[i]];
			}
		} else {
			int i = 0;
			int j = 0;
			while( i < vector1.Size && j < vector2.Size ) {
				if( vector1.Indexes[i] == vector2.Indexes[j] ) {
					sum += static_cast<double>( vector1.Values[i] ) * vector2.Values[j];
					i++;
					j++;
				} else {
					if( vector1.Indexes[i] < vector2.Indexes[j] ) {
						i++;
					} else {
						j++;
					}
				}
			}
		}
	}
	return sum;
}

// The dot product of two vectors
inline double DotProduct( const CFloatVector& vector1, const CFloatVector& vector2 )
{
	NeoPresume( vector1.Size() == vector2.Size() );
	return DotProduct( vector1.GetDesc(), vector2.GetDesc() );
}

// The dot product of two vectors
inline double DotProduct( const CFloatVector& vector1, const CSparseFloatVectorDesc& vector2 )
{
	return DotProduct( vector1.GetDesc(), vector2 );
}

// The dot product of two vectors
inline double DotProduct( const CFloatVector& vector1, const CSparseFloatVector& vector2 )
{
	return DotProduct( vector1, vector2.GetDesc() );
}

// The dot product of two vectors
inline double DotProduct( const CSparseFloatVector& vector1, const CFloatVector& vector2 )
{
	return DotProduct( vector2, vector1.GetDesc() );
}

// The dot product of two vectors
inline double DotProduct( const CSparseFloatVector& vector1, const CSparseFloatVector& vector2 )
{
	return DotProduct( vector1.GetDesc(), vector2.GetDesc() );
}

inline double LinearFunction( const CFloatVector& vector1, const CFloatVector& vector2 )
{
	NeoPresume( vector1.Size() != vector2.Size() );

	int size = 0;
	const float* operand1 = 0;
	const float* operand2 = 0;

	if( vector1.Size() == vector2.Size() + 1 ) {
		operand1 = vector1.GetPtr();
		operand2 = vector2.GetPtr();
		size = vector2.Size();
	} else {
		NeoPresume( vector1.Size() + 1 == vector2.Size() );
		operand1 = vector2.GetPtr();
		operand2 = vector1.GetPtr();
		size = vector1.Size();
	}

	double sum = operand1[size];
	for( int i = 0; i < size; ++i ) {
		sum += static_cast<double>( operand1[i] ) * operand2[i];
	}

	return sum;
}

inline double LinearFunction( const CFloatVector& vector1, const CSparseFloatVectorDesc& vector2 )
{
	NeoAssert( vector1.Size() > 0 );

	return vector1[vector1.Size() - 1] + DotProduct( vector1, vector2 );
}

inline double LinearFunction( const CFloatVector& vector1, const CSparseFloatVector& vector2 )
{
	return LinearFunction( vector1, vector2.GetDesc() );
}

inline double LinearFunction( const CSparseFloatVector& vector1, const CFloatVector& vector2 )
{
	return LinearFunction( vector2, vector1.GetDesc() );
}

// Writing into a CTextStream
inline CTextStream& operator<<( CTextStream& stream, const CFloatVector& vector )
{
	stream << "( ";
	if( vector.Size() == 0 ) {
		stream << "empty";
	} else {
		stream << vector[0];
		for( int i = 1; i < vector.Size(); i++ ) {
			stream << ", ";
			stream << vector[i];	
		}
	}
	stream << " )";
	return stream;
}

inline void CFloatVector::SetAt( int i, float what )
{
	NeoPresume( i >= 0 && i < body->Values.Size() );
	body.CopyOnWrite()->Values[i] = what;
}

inline CFloatVector operator + ( const CFloatVector& vector1, const CFloatVector& vector2 ) 
{
	CFloatVector result = vector1;
	result += vector2;
	return result;
}

inline CFloatVector operator - ( const CFloatVector& vector1, const CFloatVector& vector2 ) 
{
	CFloatVector result = vector1;
	result -= vector2;
	return result;
}

inline CFloatVector operator * ( const CFloatVector& vector, double factor ) 
{
	CFloatVector result = vector;
	result *= factor;
	return result;
}

inline CFloatVector operator * ( double factor, const CFloatVector& vector ) 
{
	return vector * factor;
}

inline CFloatVector operator / ( const CFloatVector& vector, double factor ) 
{
	return vector * (1 / factor);
}

static inline bool operator == ( const CFloatVector& vec1, const CFloatVector& vec2 )
{
	NeoAssert( vec1.Size() == vec2.Size() );

	const float* ptr1 = vec1.GetPtr();
	const float* ptr2 = vec2.GetPtr();

	for( int i = 0; i < vec1.Size(); i++ ) {
		if( ptr1[i] != ptr2[i] ) {
			return false;
		}
	}

	return true;
}

static inline bool operator != ( const CFloatVector& vec1, const CFloatVector& vec2 )
{
	return !( vec1 == vec2 );
}

inline CArchive& operator << ( CArchive& archive, const CFloatVector& vector )
{
	NeoPresume( archive.IsStoring() );

	const int size = vector.IsNull() ? NotFound : vector.Size();
	archive.WriteSmallValue( size );
	if( size > 0 ) {
		for( int i = 0; i < size; i++ ) {
			// Currently double for format reasons
			archive << static_cast<double>( vector[i] );
		}
	}

	return archive;
}

inline CArchive& operator >> ( CArchive& archive, CFloatVector& vector )
{
	NeoPresume( archive.IsLoading() );

	const int size = archive.ReadSmallValue();
	if( size == NotFound ) {
		vector = CFloatVector();
		return archive;
	}

	if( size < 0 ) {
		check( false, ERR_BAD_ARCHIVE, archive.Name() );
	}

	auto newBody = FINE_DEBUG_NEW CFloatVector::CFloatVectorBody( size );
	for( int i = 0; i < size; i++ ) {
		// Currently double for format reasons
		double temp = 0;
		archive >> temp;
		newBody->Values[i] = static_cast<float>( temp );
	}
	vector.body = newBody;

	return archive;
}

} // namespace NeoML
