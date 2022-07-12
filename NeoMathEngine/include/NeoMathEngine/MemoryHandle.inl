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

#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

template<class T>
inline void CTypedMemoryHandle<T>::SetValueAt( int index, T value ) const
{
	CTypedMemoryHandle<T> result = *this + index;
	mathEngine->DataExchangeRaw( result, &value, sizeof( T ) );
}

template<class T>
inline T CTypedMemoryHandle<T>::GetValueAt( int index ) const
{
	char result[sizeof(T)];
	CTypedMemoryHandle<T> source = *this + index;
	mathEngine->DataExchangeRaw( result, source, sizeof( T ) );
	T* value = reinterpret_cast<T*>( &result );
	return *value;
}

template<class T>
inline void CTypedMemoryHandle<T>::SetValue( T value ) const
{
	mathEngine->DataExchangeRaw( *this, &value, sizeof( T ) );
}

template<class T>
inline T CTypedMemoryHandle<T>::GetValue() const
{
	char result[sizeof(T)];
	mathEngine->DataExchangeRaw( result, *this, sizeof( T ) );
	T* value = reinterpret_cast<T*>( &result );
	return *value;
}

//------------------------------------------------------------------------------------------------------------
// CMemoryHandleVar is a variable or fixed-size array for a math engine

template<class T>
inline void CMemoryHandleVarBase<T>::SetValueAt( int index, T value )
{
	data.SetValueAt( index, value );
}

template<class T>
inline T CMemoryHandleVarBase<T>::GetValueAt( int index ) const
{
	return data.GetValueAt( index );
}

template<class T>
inline void CMemoryHandleVarBase<T>::SetValue( T value )
{
	data.SetValue( value );
}

template<class T>
inline T CMemoryHandleVarBase<T>::GetValue() const
{
	return data.GetValue();
}

//------------------------------------------------------------------------------------------------------------
// A variable or array

template<class T>
inline CMemoryHandleVar<T>::CMemoryHandleVar( IMathEngine& mathEngine, size_t size ) :
	CMemoryHandleVarBase<T>( mathEngine, size )
{
	if( size != 0 ) {
		CMemoryHandleVarBase<T>::data = CMemoryHandleVarBase<T>::mathEngine->template HeapAllocTyped<T>( size );
	}
}

template<class T>
inline CMemoryHandleVar<T>::~CMemoryHandleVar()
{
	if( !CMemoryHandleVarBase<T>::data.IsNull() ) {
		CMemoryHandleVarBase<T>::mathEngine->HeapFree( CMemoryHandleVarBase<T>::data );
	}
}

//------------------------------------------------------------------------------------------------------------

template<class T>
inline CMemoryHandleStackVar<T>::CMemoryHandleStackVar( IMathEngine& mathEngine, size_t size ) :
	CMemoryHandleVarBase<T>( mathEngine, size )
{
	if( size != 0 ) {
		CMemoryHandleVarBase<T>::data =
			CTypedMemoryHandle<T>( CMemoryHandleVarBase<T>::mathEngine->StackAlloc( size * sizeof(T) ) );
	}
}

template<class T>
inline CMemoryHandleStackVar<T>::~CMemoryHandleStackVar()
{
	if( !CMemoryHandleVarBase<T>::data.IsNull() ) {
		CMemoryHandleVarBase<T>::mathEngine->StackFree( CMemoryHandleVarBase<T>::data );
	}
}

} // namespace NeoML
