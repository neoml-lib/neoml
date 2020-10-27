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

namespace NeoML {

// Sparse vector descriptor
struct NEOML_API CSparseFloatVectorDesc {
	int Size;
	int* Indexes;
	float* Values;

	CSparseFloatVectorDesc() : Size( 0 ), Indexes( nullptr ), Values( nullptr ) {}

	static CSparseFloatVectorDesc Empty;
};

inline bool GetValue( const CSparseFloatVectorDesc& vector, int index, float& value )
{
	const int pos = FindInsertionPoint<int, Ascending<int>, int>( index, vector.Indexes, vector.Size ) - 1;
	if( pos >= 0 && vector.Indexes[pos] == index ) {
		value = vector.Values[pos];
		return true;
	}
	value = 0;
	return false;
}

inline float GetValue( const CSparseFloatVectorDesc& vector, int index )
{
	float value = 0.f;
	if( GetValue( vector, index, value ) ) {
		return value;
	}
	return 0.f;
}

//---------------------------------------------------------------------------------------------------------

// A sparse vector
// Any value that is not specified is 0
class NEOML_API CSparseFloatVector {
	static const int InitialBufferSize = 32;
public:
	CSparseFloatVector();
	explicit CSparseFloatVector( int bufferSize );
	explicit CSparseFloatVector( const CSparseFloatVectorDesc& desc );
	CSparseFloatVector( const CSparseFloatVector& other );

	CSparseFloatVectorDesc* CopyOnWrite() { return body == nullptr ? nullptr : &body.CopyOnWrite()->Desc; }
	const CSparseFloatVectorDesc& GetDesc() const { return body == nullptr ? CSparseFloatVectorDesc::Empty : body->Desc; }

	int NumberOfElements() const { return body == nullptr ? 0 : body->Desc.Size; }

	double Norm() const;
	double NormL1() const;
	float MaxAbs() const;

	void SetAt( int index, float value );
	bool GetValue( int index, float& value ) const;
	float GetValue( int index ) const;

	// Sets all values to 0
	void Nullify();

	CSparseFloatVector& operator = ( const CSparseFloatVector& vector );
	CSparseFloatVector& operator += ( const CSparseFloatVector& vector );
	CSparseFloatVector& operator -= ( const CSparseFloatVector& vector );
	CSparseFloatVector& operator *= ( double factor );
	CSparseFloatVector& operator /= ( double factor ) { return *this *= (1 / factor); }

	CSparseFloatVector& MultiplyAndAdd( const CSparseFloatVector& vector, double factor );

	// Elementwise operations
	void SquareEachElement();
	void MultiplyBy( const CSparseFloatVector& factor );
	void DivideBy( const CSparseFloatVector& divisor );

	void Serialize( CArchive& archive );

	struct CConstElement {
		const int Index;
		const float Value;
	};

	struct CModifiableElement {
		int& Index;
		float& Value;
	};
	
	class CConstIterator {
	public:
		CConstIterator( int* indexes, float* values ) : Indexes( indexes ), Values( values ) {}

		CConstElement operator*() const { return CConstElement( { *Indexes, *Values } ); }

		bool operator==( const CConstIterator& other ) const { return other.Indexes == Indexes && other.Values == Values; }
		bool operator!=( const CConstIterator& other ) const { return !( *this == other ); }
		bool operator<( const CConstIterator& other ) const { return Indexes < other.Indexes && Values < other.Values; }
		bool operator>( const CConstIterator& other ) const { return other < *this; }
		bool operator<=( const CConstIterator& other ) const { return !( *this > other ); }
		bool operator>=( const CConstIterator& other ) const { return !( *this < other ); }

		CConstIterator& operator++() { Indexes++; Values++; return *this; }
		CConstIterator operator++( int ) { CConstIterator old( *this ); Indexes++; Values++; return old; }
		CConstIterator& operator--() { Indexes--; Values--; return *this; }
		CConstIterator operator--( int ) { CConstIterator old( *this ); Indexes--; Values--; return old; }
		CConstIterator& operator+=( int offset ) { Indexes += offset; Values += offset; return *this; }
		CConstIterator operator+( int offset ) const { CConstIterator tmp( *this ); tmp += offset; return tmp; }
		CConstIterator& operator-=( int offset ) { Indexes -= offset; Values -= offset; return *this; }
		CConstIterator operator-( int offset ) const { CConstIterator tmp( *this ); tmp -= offset; return tmp; }
		int operator-( const CConstIterator& other ) const { return static_cast<int>( Indexes - other.Indexes ); }

	protected:
		int* Indexes;
		float* Values;
	};

	class CIterator : public CConstIterator {
	public:
		CIterator( int* indexes, float* values ) : CConstIterator( indexes, values ) {}

		CModifiableElement operator*() const { return CModifiableElement( { *Indexes, *Values } ); }

		CIterator& operator++() { Indexes++; Values++; return *this; }
		CIterator operator++( int ) { CIterator old( *this ); Indexes++; Values++; return old; }
		CIterator& operator--() { Indexes--; Values--; return *this; }
		CIterator operator--( int ) { CIterator old( *this ); Indexes--; Values--; return old; }
		CIterator& operator+=( int offset ) { Indexes += offset; Values += offset; return *this; }
		CIterator operator+( int offset ) const { CIterator tmp( *this ); tmp += offset; return tmp; }
		CIterator& operator-=( int offset ) { Indexes -= offset; Values -= offset; return *this; }
		CIterator operator-( int offset ) const { CIterator tmp( *this ); tmp -= offset; return tmp; }
		int operator-( const CIterator& other ) const { return static_cast<int>( Indexes - other.Indexes ); }
	};

	CConstIterator begin() const;
	CConstIterator end() const;
	CIterator begin();
	CIterator end();

private:
	// The vector body, that is, the object that stores all its data.
	struct NEOML_API CSparseFloatVectorBody : public IObject {
		const int BufferSize;
		CSparseFloatVectorDesc Desc;
		// Memory holders
		CArray<int> IndexesBuf;
		CArray<float> ValuesBuf;

		explicit CSparseFloatVectorBody( int bufferSize );
		explicit CSparseFloatVectorBody( const CSparseFloatVectorDesc& desc );
		~CSparseFloatVectorBody() override = default;

		CSparseFloatVectorBody* Duplicate() const;
	};

	CCopyOnWritePtr<CSparseFloatVectorBody> body; // The vector body.
};

// Writing into a CTextStream
inline CTextStream& operator<<( CTextStream& stream, const CSparseFloatVector& vector )
{
	const CSparseFloatVectorDesc& desc = vector.GetDesc();
	stream << "( ";
	if( desc.Size == 0 ) {
		stream << "empty";
	} else {
		stream << desc.Indexes[0] << ": " << desc.Values[0];
		for( int i = 1; i < desc.Size; i++ ) {
			stream << ", ";
			stream << desc.Indexes[i] << ": " << desc.Values[i];
		}
	}
	stream << " )";
	return stream;
}

inline CArchive& operator << ( CArchive& archive, const CSparseFloatVector& vector )
{
	NeoPresume( archive.IsStoring() );
	const_cast<CSparseFloatVector&>(vector).Serialize( archive );
	return archive;
}

inline CArchive& operator >> ( CArchive& archive, CSparseFloatVector& vector )
{
	NeoPresume( archive.IsLoading() );
	vector.Serialize( archive );
	return archive;
}

inline CSparseFloatVector::CConstIterator CSparseFloatVector::begin() const
{
	if( body == nullptr ) {
		return CConstIterator( nullptr, nullptr );
	} else {
		return CConstIterator( body->Desc.Indexes, body->Desc.Values );
	}
}

inline CSparseFloatVector::CConstIterator CSparseFloatVector::end() const
{
	if( body == nullptr ) {
		return CConstIterator( nullptr, nullptr );
	} else {
		return CConstIterator( body->Desc.Indexes + body->Desc.Size, body->Desc.Values + body->Desc.Size );
	}
}

inline CSparseFloatVector::CIterator CSparseFloatVector::begin()
{
	if( body == nullptr ) {
		return CIterator( nullptr, nullptr );
	} else {
		body.CopyOnWrite();
		return CIterator( body->Desc.Indexes, body->Desc.Values );
	}
}

inline CSparseFloatVector::CIterator CSparseFloatVector::end()
{
	if( body == nullptr ) {
		return CIterator( nullptr, nullptr );
	} else {
		body.CopyOnWrite();
		return CIterator( body->Desc.Indexes + body->Desc.Size, body->Desc.Values + body->Desc.Size );
	}
}

} // namespace NeoML
