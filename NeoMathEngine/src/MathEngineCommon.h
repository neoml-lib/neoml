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

#include <thread>
#include <mutex>
#include <unordered_map>
#include <NeoMathEngine/NeoMathEngine.h>
#include <MathEngineAllocator.h>

namespace NeoML {

inline int Ceil( int val, int discret )
{
	PRESUME_EXPR( discret > 0 );
	if( val > 0 ) {
		return ( val + discret - 1 ) / discret;
	}
	return val / discret;
}

inline int Floor( int val, int discret )
{
	PRESUME_EXPR( discret > 0 );
	if( val > 0 ) {
		return val / discret;
	}
	return ( val - discret + 1 ) / discret;
}

inline int FloorTo( int val, int discret )
{
	return Floor( val, discret ) * discret;
}

template <typename Key, typename Value>
using unordered_map = std::unordered_map<Key, Value, 
	std::hash<Key>, std::equal_to<Key>, CrtAllocator< std::pair<Key const, Value>>>;

template <typename T>
using vector = std::vector<T, CrtAllocator<T>>;

using DeleterType = void(*)(void*);

void SetThreadData( const void* key, void* data, DeleterType deleter );
void* GetThreadData( const void* key );
void CleanThreadData( const void* key );

template <typename T>
class CThreadDataPtr
{
public:
	CThreadDataPtr() = default;
	~CThreadDataPtr() { CleanThreadData( this ); }

	CThreadDataPtr( const CThreadDataPtr& ) = delete;
	CThreadDataPtr& operator=( const CThreadDataPtr& ) = delete;

	T* operator->() const { return Get(); }
	T& operator*() const { return *Get(); }

	operator bool() const 
	{  
		return GetThreadData( this ) != nullptr;
	}

	T* Get() const
	{
		return static_cast<T*>( GetThreadData( this ) );
	}

	void Reset( T* value = nullptr )
	{
		const T* currentValue = Get();
		if( value != currentValue ) {
			auto deleter = []( void* data ) { delete static_cast<T*>( data ); };
			SetThreadData( this, value, deleter );
		}
	}
};

}