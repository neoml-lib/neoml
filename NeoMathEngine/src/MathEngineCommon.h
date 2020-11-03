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

#include <cassert>
#include <new>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <NeoMathEngine/NeoMathEngine.h>
#include <MathEngineAllocator.h>

#define __merge__2( a, b )	a##b
#define __merge__1( a, b )	__merge__2( a, b )
#define __UNICODEFILE__	__merge__1( L, __FILE__ )

inline void generateAssert( NeoML::IMathEngineExceptionHandler* exceptionHandler, const char* expr, const wchar_t* file, int line, int errorCode )
{
	exceptionHandler->OnAssert( expr, file, line, errorCode );
}

inline void generateMemoryError( NeoML::IMathEngineExceptionHandler* exceptionHandler )
{
	exceptionHandler->OnMemoryError();
}

#define ASSERT_ERROR_CODE( expr ) \
	do { \
		int _err_ = (int)(expr); \
		if(_err_ != 0) { \
			NeoML::IMathEngineExceptionHandler* exceptionHandler = GetMathEngineExceptionHandler(); \
			if( exceptionHandler != 0 ) { \
				generateAssert( exceptionHandler, #expr, __UNICODEFILE__, __LINE__, _err_ ); \
			} else { \
				throw std::logic_error( #expr ); \
			} \
		} \
	} while(0)

#define ASSERT_EXPR( expr ) \
	do { \
		if(!(expr)) { \
			NeoML::IMathEngineExceptionHandler* exceptionHandler = GetMathEngineExceptionHandler(); \
			if( exceptionHandler != 0 ) { \
				generateAssert( exceptionHandler, #expr, __UNICODEFILE__, __LINE__, 0 ); \
			} else { \
				throw std::logic_error( #expr ); \
			} \
		} \
	} while(0)

#define THROW_MEMORY_EXCEPTION \
	do { \
		NeoML::IMathEngineExceptionHandler* exceptionHandler = GetMathEngineExceptionHandler(); \
		if( exceptionHandler != 0 ) { \
			generateMemoryError( exceptionHandler ); \
		} else { \
			throw std::bad_alloc(); \
		} \
	} while(0)

namespace NeoML {

inline int Ceil( int val, int discret )
{
	assert( discret > 0 );
	if( val > 0 ) {
		return ( val + discret - 1 ) / discret;
	}
	return val / discret;
}

inline int Floor( int val, int discret )
{
	assert( discret > 0 );
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

template <typename T>
class CThreadData
{
public:
	CThreadData() = default;
	~CThreadData() noexcept = default;

	CThreadData( const CThreadData& ) = delete;
	CThreadData& operator=( const CThreadData& ) = delete;

	T* Get() const
	{
		auto id = std::this_thread::get_id();

		std::lock_guard<std::mutex> lock( mutex );
		auto iterator = data.find( id );
		return ( iterator == data.end() ) ? nullptr : iterator->second.get();
	}

	T* Set( T* value )
	{
		auto id = std::this_thread::get_id();
		std::unique_ptr<T> ptr( value );
		std::lock_guard<std::mutex> lock( mutex );
		auto success = data.emplace( id, std::move( ptr ) ).second;
		assert( success == true );
		return value;
	}

private:
	mutable std::mutex mutex;
	unordered_map< std::thread::id, std::unique_ptr<T> > data;
};

}