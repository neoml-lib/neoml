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
#include <cstddef>
#include <type_traits>

namespace NeoML {

class IMathEngine;
class CMemoryHandleInternal;

// Wraps the pointer to memory allocated by a math engine
// IMPORTANT: Do not use pointers to CMemoryHandle for children classes with fields, because of the non virtual dtor.
class NEOMATHENGINE_API CMemoryHandle {
public:
	constexpr CMemoryHandle() = default;
	// Be copied and moved by default
	
	bool operator!=( const CMemoryHandle& other ) const { return !( *this == other ); }
	bool operator==( const CMemoryHandle& other ) const
		{ return MathEngine == other.MathEngine && Object == other.Object && Offset == other.Offset; }

	bool IsNull() const { return *this == CMemoryHandle{}; }

	IMathEngine* GetMathEngine() const { return MathEngine; }

protected:
	IMathEngine* MathEngine = nullptr; // the math engine owner
	const void* Object = nullptr; // the memory allocated base pointer
	std::ptrdiff_t Offset = 0; // the offset in the memory allocated volume, in bytes

	friend class CMemoryHandleInternal;

	explicit CMemoryHandle( IMathEngine* mathEngine, const void* object, ptrdiff_t offset ) :
		MathEngine( mathEngine ), Object( object ), Offset( offset ) {}

	CMemoryHandle Copy( ptrdiff_t shift ) const { return CMemoryHandle( MathEngine, Object, Offset + shift ); }
};

//---------------------------------------------------------------------------------------------------------------------

// Wraps the typed pointer to memory allocated by a math engine
// IMPORTANT: Do not use pointers to CMemoryHandle for children classes with fields, because of the non virtual dtor.
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
// IMPORTANT: Do not use pointers to CMemoryHandleVarBase for children with fields, because of the non virtual dtor.
template<class T>
class CMemoryHandleVarBase {
public:
	// Moveable only
	CMemoryHandleVarBase( CMemoryHandleVarBase&& other ) : Data( other.Data ), DataSize( other.DataSize )
		{ other.Data = CTypedMemoryHandle<T>{}; } // nullify to avoid double free
	CMemoryHandleVarBase& operator=( CMemoryHandleVarBase&& other )
		{ if( this != &other ) { std::swap( *this, other ); } return *this; }

	void SetValueAt( int index, T value ) { Data.SetValueAt( index, value ); }
	T GetValueAt( int index ) const { return Data.GetValueAt( index ); }
	void SetValue( T value ) { Data.SetValue( value ); }
	T GetValue() const { return Data.GetValue(); }

	const CTypedMemoryHandle<T>& GetHandle() const { return Data; }

	// Operators for easier use
	operator const CTypedMemoryHandle<T>&() const { return GetHandle(); }
	operator CTypedMemoryHandle<const T>() const { return GetHandle(); }
	CTypedMemoryHandle<T> operator []( int index ) const { return GetHandle() + index; }

	bool operator==( const CTypedMemoryHandle<const T>& other ) const { return GetHandle() == other; }
	bool operator!=( const CTypedMemoryHandle<const T>& other ) const { return !( *this == other ); }

	CTypedMemoryHandle<T> operator+( ptrdiff_t shift ) const { return GetHandle() + shift; }
	CTypedMemoryHandle<T> operator-( ptrdiff_t shift ) const { return GetHandle() - shift; }
	int operator-( const CTypedMemoryHandle<T>& handle ) const { return GetHandle() - handle; }

	int Size() const { return static_cast<int>( DataSize ); }
	IMathEngine* GetMathEngine() const { return Data.GetMathEngine(); }

protected:
	CTypedMemoryHandle<T> Data; // the typed memory handler
	const size_t DataSize; // the typed memory size

	explicit CMemoryHandleVarBase( size_t size ) : DataSize( size ) {}
	~CMemoryHandleVarBase() = default;

private:
	// may not be copied, only passed by reference
	CMemoryHandleVarBase( const CMemoryHandleVarBase& ) = delete;
	CMemoryHandleVarBase& operator=( const CMemoryHandleVarBase& ) = delete;
};

//---------------------------------------------------------------------------------------------------------------------

// A variable or an array on the heap
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
