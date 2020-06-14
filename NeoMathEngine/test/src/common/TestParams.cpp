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

#include <NeoMathEngine/Platforms.h>
#include "TestParams.h"

namespace NeoMLTest {

bool Value( const std::string& value, int& result )
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

bool Value( const std::string& str, double& result )
{
	std::string tempStr = str;
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

//------------------------------------------------------------------------------------------------------------

static size_t strrspn( const char* str, const char* symbols )
{
	int len = static_cast<int>(strlen( str ));
	int symbolsCount = static_cast<int>(strlen( symbols ));
	int res = 0;

	for( int i = len - 1; i >= 0; i-- ) {
		bool found = false;
		for( int j = 0; j < symbolsCount; j++ ) {
			if( str[i] == symbols[j] ) {
				found = true;
				break;
			}
		}
		if( !found ) {
			break;
		}
		res++;
	}
	return static_cast<size_t>(res);
}

static int find( const char* str, char symbol )
{
	const char* ptr = strchr( str, symbol );
	if( ptr == 0 ) {
		return -1;
	}
	return static_cast<int>( ptr - str );
}

static int findStr( const char* str, const char* substr )
{
	const char* ptr = strstr( str, substr );
	if( ptr == 0 ) {
		return -1;
	}
	return static_cast<int>( ptr - str );
}

static std::string trim( const std::string& str )
{
	size_t leftSize = strspn( str.c_str(), " \t\r\n\f\v" );
	size_t rightSize = strrspn( str.c_str(), " \t\r\n\f\v" );
	const char* res = str.c_str();
	size_t len = strlen( res );
	return std::string( res + leftSize, static_cast<int>( len - leftSize - rightSize ) );
}

CTestParams::CTestParams( const std::string& string )
{
	std::vector<std::string> keyValue;
	splitStringsByDelimiter( keyValue, string, ";" );

	for( size_t i = 0; i < keyValue.size(); ++i ) {
		const int equalSign = find( keyValue[i].c_str(), '=' );
		const char* ptr = keyValue[i].c_str();
		std::string key = std::string( ptr, equalSign );
		key = trim( key );
		std::string value = std::string( ptr + equalSign + 1 );
		value = trim( value );
		flags.insert(make_pair( key, value ));
	}
}

CTestParams::CTestParams( const CTestParams& other ) :
	flags( other.flags )
{
}

std::string CTestParams::GetStrValue( const std::string& key ) const
{
	return flags.find( key )->second;
}

CTestParams& CTestParams::operator=( const CTestParams& other )
{
	flags = other.flags;
	return *this;
}

CInterval CTestParams::GetInterval( const std::string& key ) const
{
	return parseInterval( flags.find( key )->second );
}

CInterval CTestParams::parseInterval( const std::string& valueString ) const
{
	const int openBracket = find( valueString.c_str(), '(' );

	if( openBracket < 0 ) {
		CInterval result;
		Value( valueString, result.Begin );
		result.End = result.Begin;
		return result;
	}

	const char* valueStringPtr = valueString.c_str();
	const size_t valueStringLen = strlen( valueStringPtr );
	const size_t closeBracket = find( valueStringPtr, ')' );
	const int firstDot = findStr( valueStringPtr, ".." );
	unsigned int dotsEnd = firstDot + 2;
	while( dotsEnd < valueStringLen && valueString[dotsEnd] == '.' ) {
		++dotsEnd;
	}

	CInterval result;
	Value( std::string( valueStringPtr + openBracket + 1, static_cast<int>( firstDot - openBracket - 1 ) ), result.Begin );
	Value( std::string( valueStringPtr + dotsEnd, static_cast<int>( closeBracket - dotsEnd ) ), result.End );
	return result;
}

::std::ostream& operator<<( ::std::ostream& os, const CTestParams& params )
{
	os << "\n";
	const std::unordered_map<std::string, std::string>& flags = params.flags;
	for( auto i : flags ) {
		os << i.first << " " << i.second << "\n";
	}
	return os;
}

void CTestParams::splitStringsByDelimiter( std::vector<std::string>& result, const std::string& string, const std::string& delimiter ) const
{
	result.clear();
	const char* stringPtr = string.c_str();
	const unsigned int stringLen = static_cast<unsigned int>( string.length() );
	const unsigned int delimiterLen = static_cast<unsigned int>( delimiter.length() );
	for( unsigned int pos = 0; pos <= stringLen; ) {
		int nextDelimeter = findStr( stringPtr + pos, delimiter.c_str() );
		if( nextDelimeter < 0 ) {
			nextDelimeter = stringLen - pos;
		}
		if( nextDelimeter > 0 ) {
			result.push_back( std::string( stringPtr + pos, nextDelimeter ) );
		}
		pos += static_cast<unsigned int>(nextDelimeter) + delimiterLen;
	}
}

} // namespace NeoMLTest
