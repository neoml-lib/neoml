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

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <cstddef>
#include <type_traits>

namespace NeoML {

class IMathEngine;
class CMemoryHandleInternal;

// Wraps the pointer to memory allocated by a math engine
class NEOMATHENGINE_API CMemoryHandle {
public:
	CMemoryHandle() : mathEngine( 0 ), object( 0 ), offset( 0 ) {}
	CMemoryHandle( const CMemoryHandle& other ) : mathEngine( other.mathEngine ), object( other.object ), offset( other.offset ) {}

	CMemoryHandle& operator=( const CMemoryHandle& other )
	{
		mathEngine = other.mathEngine;
		object = other.object;
		offset = other.offset;
		return *this;
	}

	bool operator==( const CMemoryHandle& other ) const
	{
		return mathEngine == other.mathEngine && object == other.object && offset == other.offset;
	}

	bool operator!=( const CMemoryHandle& other ) const
	{
		return !operator==( other );
	}

	bool IsNull() const
	{
		return mathEngine == 0 && object == 0 && offset == 0;
	}

	IMathEngine* GetMathEngine() const { return mathEngine; }

protected:
	IMathEngine* mathEngine; // the math engine
	const void* object; // the base object
	std::ptrdiff_t offset; // the offset in the base object, in bytes

	friend class CMemoryHandleInternal;

	explicit CMemoryHandle( IMathEngine* _mathEngine, const void* _object, ptrdiff_t _offset ) : mathEngine( _mathEngine ), object( _object ), offset( _offset ) {}

	CMemoryHandle CopyMemoryHandle( ptrdiff_t shift ) const { return CMemoryHandle( mathEngine, object, offset + shift ); }
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
