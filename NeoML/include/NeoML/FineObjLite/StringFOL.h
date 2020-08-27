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

#include <cstdarg>
#include <string>

namespace FObj {

class CString : public std::string {
public:
	CString() {}
	CString( const char* str ) : std::string( str ) {}
	CString( const char* str, int len ) : std::string( str, len ) {}
	CString( const std::string& str ) : std::string( str ) {}

	operator const char*() const { return data(); }

	int Find( const CString& other ) const;
};

inline int CString::Find( const CString& other ) const
{
	size_t found = std::string::find( other );
	return found == std::string::npos ? -1 : static_cast<int>( found );
}

inline CString operator+( const CString& first, const CString& second )
{
	return CString( static_cast<const std::string&>( first ) + static_cast<const std::string&>( second ) );
}

inline CString operator+( const CString& first, const char* second )
{
	return first + CString( second );
}

inline CString operator+( const char* first, const CString& second )
{
	return CString( first ) + second;
}

//------------------------------------------------------------------------------------------------------------

inline CString SubstParam( const char* text, const char* params[], int size )
{
	const int MaxIndexLen = 8;

	CString ret;
	size_t len = strlen( text );
	size_t pos = 0;
	const char* ptr = text;
	while( pos < len ) {
		const char* percentPtr = ::strchr( ptr + pos, '%' );
		if( percentPtr == 0 ) {
			ret += CString( ptr + pos );
			break;
		}
		const size_t percentPtrOffset = percentPtr - ( ptr + pos );

		ret += std::string( ptr + pos, percentPtrOffset );
		pos += percentPtrOffset;
		size_t digitPos = pos + 1;
		int indexValue = 0;
		int indexLen = 0;
		for( ; digitPos < len && isdigit( ptr[digitPos] ); digitPos++ ) {
			indexValue = indexValue * 10 + ( ptr[digitPos] - '0' );
			indexLen++;
		}
		if( 0 <= indexValue && indexValue < size && indexLen < MaxIndexLen ) {
			ret += params[indexValue];
		} else {
			ret += std::string( ptr + pos, digitPos - pos );
		}
		pos = digitPos;
	}
	return ret;
}

template <typename T>
inline CString Str( T value )
{
	return std::to_string( value );
}

inline bool Value( const CString& value, int& result )
{
	const char* ptr = value.data() + strspn( value.data(), " \t\r\n\f\v" );

	char* endPtr = 0;
	errno = 0;
	int tmp = static_cast<int>( ::strtol( ptr, &endPtr, 10 ) );
	if( errno == ERANGE || endPtr == ptr ) {
		return false;
	}
	int pos = static_cast<int>( endPtr - value.data() );
	if( pos + strspn( value.data() + pos, " \t\r\n\f\v" ) != value.length() ) {
		return false;
	}
	result = tmp;
	return true;
}

inline bool Value( const CString& str, double& result )
{
	CString tempStr = str;
	char* strPtr = const_cast<char*>( tempStr.data() + strspn( tempStr.data(), " \t\r\n\f\v" ) );
	int strLen = static_cast<int>( strlen( tempStr.data() ) - ( strPtr - tempStr.data() ) );

#if FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_IOS ) || FINE_PLATFORM( FINE_DARWIN )
	char decimalPoint = localeconv()->decimal_point[0];

	for( int i = 0; i < strLen; i++ ) {
		if( strPtr[i] == '.' ) {
			strPtr[i] = decimalPoint;
		}
		if( strPtr[i] == ',' ) {
			strPtr[i] = decimalPoint;
		}
	}

	char* endPtr = 0;
	errno = 0;
	double tmp = ::strtod( strPtr, &endPtr );
	if( endPtr != strPtr && *endPtr == '\0' && errno != ERANGE ) {
		result = tmp;
		return true;
	}
	return false;
#elif FINE_PLATFORM( FINE_ANDROID )
	for( int i = 0; i < strLen; i++ ) {
		if( strPtr[i] == ',' ) {
			strPtr[i] = '.';
		}
	}

	char* endPtr = 0;
	errno = 0;
	double tmp = ::strtod( strPtr, &endPtr );
	if( endPtr != str.c_str() && *endPtr == '\0' && errno != ERANGE ) {
		result = tmp;
		return true;
	}

	for( int i = 0; i < strLen; i++ ) {
		if( strPtr[i] == '.' ) {
			strPtr[i] = ',';
		}
	}

	endPtr = 0;
	errno = 0;
	tmp = ::strtod( strPtr, &endPtr );
	if( endPtr != strPtr && *endPtr == '\0' && errno != ERANGE ) {
		result = tmp;
		return true;
	}
	return false;
#else
	#error Unknown platform
#endif
}

} // namespace FObj
