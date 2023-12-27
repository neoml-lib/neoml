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
#include <climits>
#include <type_traits>

namespace NeoML {

class IMathEngine;
class CMemoryHandleInternal;

// Get pointer to IMathEngine by the given current entity
NEOMATHENGINE_API IMathEngine* GetMathEngineByIndex( size_t currentEntity );
// Get current entity from the given pointer to IMathEngine
NEOMATHENGINE_API size_t GetIndexOfMathEngine( const IMathEngine* mathEngine );

// Wraps the pointer to memory allocated by a math engine
class NEOMATHENGINE_API CMemoryHandle {
private:
#if FINE_PLATFORM( FINE_64_BIT )
	static constexpr int mathEngineCountWidth = 10; // compress to bitfield
	static constexpr int mathEngineCountShift = ( sizeof( size_t ) * CHAR_BIT ) - mathEngineCountWidth;
	static constexpr size_t mathEngineMaxOffset = size_t( 1 ) << mathEngineCountShift;
#else  // FINE_32_BIT
	// only for bitfield compiles correct. no compress
	static constexpr int mathEngineCountWidth = sizeof( size_t ) * CHAR_BIT;
	static constexpr int mathEngineCountShift = sizeof( size_t ) * CHAR_BIT;
#endif // FINE_32_BIT

public:
	// Any possible number of all mathEngines
	static constexpr int MaxMathEngineEntities = 1024;
	static constexpr size_t MathEngineEntityInvalid = size_t( -1 );

	CMemoryHandle() : CMemoryHandle( nullptr, 0, MathEngineEntityInvalid ) {}
	// Be copied and moved by default

	bool operator!=( const CMemoryHandle& other ) const { return !operator==( other ); }
	bool operator==( const CMemoryHandle& other ) const
		{ return Object == other.Object && Offset == other.Offset && Entity == other.Entity; }

	bool IsNull() const { return *this == CMemoryHandle{}; }

	IMathEngine* GetMathEngine() const { return GetMathEngineByIndex( Entity ); }

protected:
	// struct of (16 bytes size for x64 and arm-x64) and (12 bytes size for x86 and arm-x32)
	const void* Object = nullptr; // the memory allocated base pointer
	// The offset in the memory allocated volume, in bytes
	size_t Offset : mathEngineCountShift; // (x64) the less significant bits of size_t stores offset in the base object, in bytes
	// The math engine owner
	size_t Entity : mathEngineCountWidth; // (x64) the most significant bits of size_t stores the number of IMathEngine entity

	friend class CMemoryHandleInternal;

	explicit CMemoryHandle( IMathEngine* mathEngine, const void* object, ptrdiff_t offset ) :
		CMemoryHandle( object , offset, GetIndexOfMathEngine( mathEngine ) ) {}

	CMemoryHandle Copy( ptrdiff_t shift ) const { return CMemoryHandle( Object, Offset + shift, Entity ); }

private:
	explicit CMemoryHandle( const void* object, ptrdiff_t offset, size_t entity ) :
		Object( object ), Offset( offset ), Entity( entity & ( MaxMathEngineEntities - 1 ) )
	{
#if FINE_PLATFORM( FINE_64_BIT )
		static_assert( MaxMathEngineEntities == ( 1 << mathEngineCountWidth ), "Invalid max MathEngine entities" );
		// Checks that the most significant bits do not interfere the result
		ASSERT_EXPR( 0 <= offset && size_t( offset ) < mathEngineMaxOffset );
#endif // FINE_64_BIT
		ASSERT_EXPR( entity == MathEngineEntityInvalid || entity < ( MaxMathEngineEntities - 1/*Invalid*/ ) );
	}
};

//---------------------------------------------------------------------------------------------------------------------

// Wraps the typed pointer to memory allocated by a math engine
template <class T>
class CTypedMemoryHandle : public CMemoryHandle {
public:
	constexpr CTypedMemoryHandle() = default;
	// Converting ctor
	explicit CTypedMemoryHandle( const CMemoryHandle& other ) : CMemoryHandle( other ) {}
	// Be copied and moved by default

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
		Offset += shift * sizeof( T );
		return *this;
	}

	CTypedMemoryHandle& operator-=( ptrdiff_t shift )
	{
		Offset -= shift * sizeof( T );
		return *this;
	}

	CTypedMemoryHandle& operator++()
	{
		Offset += sizeof( T );
		return *this;
	}

	CTypedMemoryHandle<T> operator++( int )
	{
		CTypedMemoryHandle result( *this );
		Offset += sizeof( T );
		return result;
	}

	CTypedMemoryHandle& operator--()
	{
		Offset -= sizeof( T );
		return *this;
	}

	CTypedMemoryHandle<const T> operator--( int )
	{
		CTypedMemoryHandle result( *this );
		Offset -= sizeof( T );
		return result;
	}

	CTypedMemoryHandle operator+( ptrdiff_t shift ) const
	{
		return CTypedMemoryHandle<T>( Copy( shift * sizeof( T ) ) );
	}

	CTypedMemoryHandle operator-( ptrdiff_t shift ) const
	{
		return operator+( -shift );
	}

	int operator-( const CTypedMemoryHandle& handle ) const
	{
		return ( int ) ( Offset - handle.Offset );
	}
};

//---------------------------------------------------------------------------------------------------------------------

// CMemoryHandleVar is a variable or a fixed-size array for a math engine
template<class T>
class CMemoryHandleVarBase {
public:
	void SetValueAt( int index, T value );
	T GetValueAt( int index ) const;
	void SetValue( T value );
	T GetValue() const;

	const CTypedMemoryHandle<T>& GetHandle() const { return Data; }

	// Operators for easier use
	operator const CTypedMemoryHandle<T>&( ) const { return GetHandle(); }
	operator CTypedMemoryHandle<const T>() const { return GetHandle(); }
	CTypedMemoryHandle<T> operator []( int index ) const { return GetHandle() + index; }

	bool operator==( const CTypedMemoryHandle<const T>& other ) const { return GetHandle() == other; }
	bool operator!=( const CTypedMemoryHandle<const T>& other ) const { return GetHandle() != other; }

	CTypedMemoryHandle<T> operator+( ptrdiff_t shift ) const { return GetHandle() + shift; }
	CTypedMemoryHandle<T> operator-( ptrdiff_t shift ) const { return GetHandle() - shift; }
	int operator-( const CTypedMemoryHandle<T>& handle ) const { return GetHandle() - handle; }

	int Size() const { return static_cast<int>( DataSize ); }
	IMathEngine* GetMathEngine() const { return &MathEngine; }

protected:
	IMathEngine& MathEngine; // the math engine owner
	mutable CTypedMemoryHandle<T> Data; // the typed memory handler
	const size_t DataSize; // the typed memory size

	CMemoryHandleVarBase( IMathEngine& mathEngine, size_t size ) : MathEngine( mathEngine ), DataSize( size ) {}
	~CMemoryHandleVarBase() = default;

private:
	// may not be copied, only passed by reference
	CMemoryHandleVarBase( const CMemoryHandleVarBase& ) = delete;
	CMemoryHandleVarBase& operator=( const CMemoryHandleVarBase& ) = delete;
};

//---------------------------------------------------------------------------------------------------------------------

// A variable or an array
template<class T>
class CMemoryHandleVar : public CMemoryHandleVarBase<T> {
public:
	explicit CMemoryHandleVar( IMathEngine& mathEngine, size_t size = 1 );

	~CMemoryHandleVar();
};

//---------------------------------------------------------------------------------------------------------------------

// A variable or an array on the stack
template<class T>
class CMemoryHandleStackVar : public CMemoryHandleVarBase<T> {
public:
	explicit CMemoryHandleStackVar( IMathEngine& mathEngine, size_t size = 1 );

	~CMemoryHandleStackVar();
};

//---------------------------------------------------------------------------------------------------------------------

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
