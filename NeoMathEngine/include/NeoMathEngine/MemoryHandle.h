/* Copyright © 2017-2024 ABBYY

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

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/NeoMathEngineException.h>
#include <cstddef>
#include <type_traits>

namespace NeoML {

class IMathEngine;
class CMemoryHandleInternal;

// Get pointer to IMathEngine by the given CurrentEntity
inline IMathEngine* GetMathEngineByIndex( size_t currentEntity );
// Get current entity from the given pointer to IMathEngine
inline size_t GetIndexOfMathEngine( const IMathEngine* mathEngine );

// Wraps the pointer to memory allocated by a math engine
class NEOMATHENGINE_API CMemoryHandle {
private:
#if FINE_PLATFORM( FINE_64_BIT )
	static constexpr int MathEngineCountWidth = 10; // compress to bitfield
	static constexpr int MathEngineCountShift = ( sizeof( size_t ) * 8 /*bits*/ ) - MathEngineCountWidth;
	static constexpr size_t MathEngineEntityInvalid = ( ( size_t( 1 ) << MathEngineCountWidth ) - 1 );
	static constexpr size_t MathEngineMaxOffset = size_t( 1 ) << MathEngineCountShift;
#else  // FINE_32_BIT
	// only for bitfield compiles correct. no compress
	static constexpr int MathEngineCountWidth = sizeof( size_t ) * 8;
	static constexpr int MathEngineCountShift = sizeof( size_t ) * 8;
	static constexpr size_t MathEngineEntityInvalid = size_t( -1 );
#endif // FINE_32_BIT

public:
	// Any possible number of all mathEngines
	static constexpr int MaxMathEngineEntities = 1024;

	CMemoryHandle() : object( nullptr ), offset( 0 ), entity( MathEngineEntityInvalid ) {}
	CMemoryHandle( const CMemoryHandle& other ) : object( other.object ), offset( other.offset ), entity( other.entity ) {}

	CMemoryHandle& operator=( const CMemoryHandle& other )
	{
		if( this != &other ) {
			object = other.object;
			offset = other.offset;
			entity = other.entity;
		}
		return *this;
	}

	bool operator==( const CMemoryHandle& other ) const
	{ return object == other.object && offset == other.offset && entity == other.entity; }

	bool operator!=( const CMemoryHandle& other ) const { return !operator==( other ); }

	bool IsNull() const { return object == nullptr && offset == 0 && entity == MathEngineEntityInvalid; }

	IMathEngine* GetMathEngine() const { return GetMathEngineByIndex( entity ); }

protected:
	// struct of (16 bytes size for x64 and arm-x64) and (12 bytes size for x86 and arm-x32)
	const void* object; // the base object
	size_t offset : MathEngineCountShift; // (x64) the less significant bits of size_t stores offset in the base object, in bytes
	size_t entity : MathEngineCountWidth; // (x64) the most significant bits of size_t stores the number of IMathEngine entity

	friend class CMemoryHandleInternal;

	explicit CMemoryHandle( IMathEngine* _mathEngine, const void* _object, ptrdiff_t _offset ) :
		CMemoryHandle( _object , _offset, GetIndexOfMathEngine( _mathEngine ) )
	{}

	CMemoryHandle CopyMemoryHandle( ptrdiff_t shift ) const { return CMemoryHandle( object, offset + shift, entity ); }

private:
	explicit CMemoryHandle( const void* _object, ptrdiff_t _offset, size_t _entity ) :
		object( _object ), offset( _offset ), entity( _entity )
	{
#if FINE_PLATFORM( FINE_64_BIT )
		static_assert( MaxMathEngineEntities == ( 1 << MathEngineCountWidth ), "Invalid MaxMathEngineEntities" );
		// Checks that the most significant bits do not interfere the result
		ASSERT_EXPR( 0 <= _offset && size_t( _offset ) < MathEngineMaxOffset );
#endif // FINE_64_BIT
		ASSERT_EXPR( _entity < MaxMathEngineEntities );
	}
};

//------------------------------------------------------------------------------------------------------------

// Wraps the typed pointer to memory allocated by a math engine
template <class T>
class CTypedMemoryHandle : public CMemoryHandle {
public:
	CTypedMemoryHandle() = default;
	explicit CTypedMemoryHandle( const CMemoryHandle& other ) : CMemoryHandle( other ) {}

	void SetValueAt( int index, T value ) const;
	T GetValueAt( int index ) const;
	void SetValue( T value ) const;
	T GetValue() const;

	template<typename U = T, typename std::enable_if<std::is_same<U, T>::value && !std::is_const<U>::value, int>::type = 0>
	operator CTypedMemoryHandle<const U>() const
	{
		return CTypedMemoryHandle<const U>( static_cast<const CMemoryHandle&>( *this ) );
	}

	CTypedMemoryHandle& operator+=( ptrdiff_t shift )
	{
		offset += shift * sizeof( T );
		return *this;
	}

	CTypedMemoryHandle& operator-=( ptrdiff_t shift )
	{
		offset -= shift * sizeof( T );
		return *this;
	}

	CTypedMemoryHandle& operator++()
	{
		offset += sizeof( T );
		return *this;
	}

	CTypedMemoryHandle<T> operator++( int )
	{
		CTypedMemoryHandle result( *this );
		offset += sizeof( T );
		return result;
	}

	CTypedMemoryHandle& operator--()
	{
		offset -= sizeof( T );
		return *this;
	}

	CTypedMemoryHandle<const T> operator--( int )
	{
		CTypedMemoryHandle result( *this );
		offset -= sizeof( T );
		return result;
	}

	CTypedMemoryHandle operator+( ptrdiff_t shift ) const
	{
		return CTypedMemoryHandle<T>( CopyMemoryHandle( shift * sizeof( T ) ) );
	}

	CTypedMemoryHandle operator-( ptrdiff_t shift ) const
	{
		return operator+( -shift );
	}

	int operator-( const CTypedMemoryHandle& handle ) const
	{
		return ( int ) ( offset - handle.offset );
	}
};

//------------------------------------------------------------------------------------------------------------

// CMemoryHandleVar is a variable or fixed-size array for a math engine
template<class T>
class CMemoryHandleVarBase {
public:
	void SetValueAt( int index, T value );
	T GetValueAt( int index ) const;
	void SetValue( T value );
	T GetValue() const;

	const CTypedMemoryHandle<T>& GetHandle() const { return data; }

	// Operators for easier use
	operator const CTypedMemoryHandle<T>&( ) const { return GetHandle(); }
	operator CTypedMemoryHandle<const T>() const { return GetHandle(); }
	CTypedMemoryHandle<T> operator []( int index ) const { return GetHandle() + index; }

	bool operator==( const CTypedMemoryHandle<const T>& other ) const { return GetHandle() == other; }
	bool operator!=( const CTypedMemoryHandle<const T>& other ) const { return GetHandle() != other; }

	CTypedMemoryHandle<T> operator+( ptrdiff_t shift ) const { return GetHandle() + shift; }

	CTypedMemoryHandle<T> operator-( ptrdiff_t shift ) const { return GetHandle() - shift; }

	int operator-( const CTypedMemoryHandle<T>& handle ) const { return GetHandle() - handle; }

	int Size() const { return static_cast<int>( size ); }

	IMathEngine* GetMathEngine() const { return mathEngine; }

protected:
	CMemoryHandleVarBase( IMathEngine& _mathEngine, size_t _size ) : mathEngine( &_mathEngine ), size( _size ) {}

	mutable IMathEngine* mathEngine;
	mutable CTypedMemoryHandle<T> data;
	const size_t size;

private:
	// may not be copied, only passed by reference
	CMemoryHandleVarBase( const CMemoryHandleVarBase& );
	CMemoryHandleVarBase& operator=( const CMemoryHandleVarBase& );
};

//------------------------------------------------------------------------------------------------------------

// A variable or array
template<class T>
class CMemoryHandleVar : public CMemoryHandleVarBase<T> {
public:
	explicit CMemoryHandleVar( IMathEngine& mathEngine, size_t size = 1 );

	~CMemoryHandleVar();

	const CTypedMemoryHandle<T>& GetHandle() const { return CMemoryHandleVarBase<T>::GetHandle(); }
};

//------------------------------------------------------------------------------------------------------------

// A variable or array on stack
template<class T>
class CMemoryHandleStackVar : public CMemoryHandleVarBase<T> {
public:
	explicit CMemoryHandleStackVar( IMathEngine& mathEngine, size_t size = 1 );

	~CMemoryHandleStackVar();
};

//------------------------------------------------------------------------------------------------------------
// typedefs
typedef CTypedMemoryHandle<float> CFloatHandle;
typedef CTypedMemoryHandle<const float> CConstFloatHandle;

typedef CTypedMemoryHandle<int> CIntHandle;
typedef CTypedMemoryHandle<const int> CConstIntHandle;

typedef CMemoryHandleVar<float> CFloatHandleVar;
typedef CMemoryHandleVar<int> CIntHandleVar;

typedef CMemoryHandleStackVar<float> CFloatHandleStackVar;
typedef CMemoryHandleStackVar<int> CIntHandleStackVar;

} // namespace NeoML
