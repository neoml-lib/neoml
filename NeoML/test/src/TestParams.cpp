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

#include <common.h>
#pragma hdrstop

#include <TestParams.h>

namespace NeoMLTest {

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
		return NotFound;
	}
	return static_cast<int>( ptr - str );
}

static int findStr( const char* str, const char* substr )
{
	const char* ptr = strstr( str, substr );
	if( ptr == 0 ) {
		return NotFound;
	}
	return static_cast<int>( ptr - str );
}

static CString trim( const CString& str )
{
	size_t leftSize = strspn( str, " \t\r\n\f\v" );
	size_t rightSize = strrspn( str, " \t\r\n\f\v" );
	const char* res = str;
	size_t len = strlen( res );
	return CString( res + leftSize, static_cast<int>( len - leftSize - rightSize ) );
}

CTestParams::CTestParams( const CString& string )
{
	CArray<CString> keyValue;
	splitStringsByDelimiter( keyValue, string, ";" );

	for( int i = 0; i < keyValue.Size(); ++i ) {
		const int equalSign = find( keyValue[i], '=' );
		NeoAssert( equalSign != NotFound );
		const char* ptr = keyValue[i];
		CString key = CString( ptr, equalSign );
		key = trim( key );
		CString value = CString( ptr + equalSign + 1 );
		value = trim( value );
		flags.Add( key, value );
	}
}

CTestParams::CTestParams( const CTestParams& other )
{
	other.flags.CopyTo( flags );
}

CString CTestParams::GetStrValue( const CString& key ) const
{
	NeoAssert( flags.Has( key ) );
	return flags.Get( key );
}

CTestParams& CTestParams::operator=( const CTestParams& other )
{
	other.flags.CopyTo( flags );
	return *this;
}

CInterval CTestParams::GetInterval( const CString& key ) const
{
	NeoAssert( flags.Has( key ) );
	return parseInterval( flags.Get( key ) );
}

CInterval CTestParams::parseInterval( const CString& valueString ) const
{
	const int openBracket = find( valueString, '(' );

	if( openBracket == NotFound ) {
		CInterval result;
		NeoAssert( Value( valueString, result.Begin ) );
		result.End = result.Begin;
		return result;
	}

	const char* valueStringPtr = valueString;
	const size_t valueStringLen = strlen( valueStringPtr );
	const size_t closeBracket = find( valueStringPtr, ')' );
	const int firstDot = findStr( valueStringPtr, ".." );
	unsigned int dotsEnd = firstDot + 2;
	while( dotsEnd < valueStringLen && valueString[dotsEnd] == '.' ) {
		++dotsEnd;
	}

	CInterval result;
	NeoAssert( Value( CString( valueStringPtr + openBracket + 1, static_cast<int>( firstDot - openBracket - 1 ) ), result.Begin ) );
	NeoAssert( Value( CString( valueStringPtr + dotsEnd, static_cast<int>( closeBracket - dotsEnd ) ), result.End ) );
	return result;
}

::std::ostream& operator<<( ::std::ostream& os, const CTestParams& params )
{
	os << "\n";
	const CMap<CString, CString>& flags = params.flags;
	for( int i = flags.GetFirstPosition(); i != NotFound; i = flags.GetNextPosition( i ) ) {
		os << flags.GetKey( i ) << " " << flags.GetValue( i ) << "\n";
	}
	return os;
}

void CTestParams::splitStringsByDelimiter( CArray<CString>& result, const CString& string, const CString& delimiter ) const
{
	NeoAssert( delimiter != "" );

	result.DeleteAll();
	const char* stringPtr = string;
	const unsigned int stringLen = static_cast<unsigned int>( strlen( string ) );
	const unsigned int delimiterLen = static_cast<unsigned int>( strlen( delimiter ) );
	for( unsigned int pos = 0; pos <= stringLen; ) {
		int nextDelimeter = findStr( stringPtr + pos, delimiter );
		if( nextDelimeter == NotFound ) {
			nextDelimeter = stringLen - pos;
		}
		if( nextDelimeter > 0 ) {
			result.Add( CString( stringPtr + pos, nextDelimeter ) );
		}
		pos += static_cast<unsigned int>(nextDelimeter) + delimiterLen;
	}
}

} // namespace NeoMLTest
