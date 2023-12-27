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

#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

template<class T>
inline void CTypedMemoryHandle<T>::SetValueAt( int index, T value ) const
{
	CTypedMemoryHandle<T> result = *this + index;
	GetMathEngine()->DataExchangeRaw( result, &value, sizeof( T ) );
}

template<class T>
inline T CTypedMemoryHandle<T>::GetValueAt( int index ) const
{
	char result[sizeof(T)];
	CTypedMemoryHandle<T> source = *this + index;
	GetMathEngine()->DataExchangeRaw( result, source, sizeof( T ) );
	T* value = reinterpret_cast<T*>( &result );
	return *value;
}

template<class T>
inline void CTypedMemoryHandle<T>::SetValue( T value ) const
{
	GetMathEngine()->DataExchangeRaw( *this, &value, sizeof( T ) );
}

template<class T>
inline T CTypedMemoryHandle<T>::GetValue() const
{
	char result[sizeof(T)];
	GetMathEngine()->DataExchangeRaw( result, *this, sizeof( T ) );
	T* value = reinterpret_cast<T*>( &result );
	return *value;
}

//------------------------------------------------------------------------------------------------------------
// CMemoryHandleVar is a variable or fixed-size array for a math engine

template<class T>
inline void CMemoryHandleVarBase<T>::SetValueAt( int index, T value )
{
	Data.SetValueAt( index, value );
}

template<class T>
inline T CMemoryHandleVarBase<T>::GetValueAt( int index ) const
{
	return Data.GetValueAt( index );
}

template<class T>
inline void CMemoryHandleVarBase<T>::SetValue( T value )
{
	Data.SetValue( value );
}

template<class T>
inline T CMemoryHandleVarBase<T>::GetValue() const
{
	return Data.GetValue();
}

//------------------------------------------------------------------------------------------------------------
// A variable or array

template<class T>
inline CMemoryHandleVar<T>::CMemoryHandleVar( IMathEngine& mathEngine, size_t size ) :
	CMemoryHandleVarBase<T>( mathEngine, size )
{
	if( size != 0 ) {
		CMemoryHandleVarBase<T>::Data = CMemoryHandleVarBase<T>::MathEngine.template HeapAllocTyped<T>( size );
	}
}

template<class T>
inline CMemoryHandleVar<T>::~CMemoryHandleVar()
{
	if( !CMemoryHandleVarBase<T>::Data.IsNull() ) {
		CMemoryHandleVarBase<T>::MathEngine.HeapFree( CMemoryHandleVarBase<T>::Data );
	}
}

//------------------------------------------------------------------------------------------------------------

template<class T>
inline CMemoryHandleStackVar<T>::CMemoryHandleStackVar( IMathEngine& mathEngine, size_t size ) :
	CMemoryHandleVarBase<T>( mathEngine, size )
{
	if( size != 0 ) {
		CMemoryHandleVarBase<T>::Data =
			CTypedMemoryHandle<T>( CMemoryHandleVarBase<T>::MathEngine.StackAlloc( size * sizeof( T ) ) );
	}
}

template<class T>
inline CMemoryHandleStackVar<T>::~CMemoryHandleStackVar()
{
	if( !CMemoryHandleVarBase<T>::Data.IsNull() ) {
		CMemoryHandleVarBase<T>::MathEngine.StackFree( CMemoryHandleVarBase<T>::Data );
	}
}

} // namespace NeoML
