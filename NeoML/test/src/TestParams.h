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
#include <common.h>

namespace NeoMLTest {

class CTestParams {
public:
	explicit CTestParams( const CString& string );
	CTestParams( const CTestParams& other );

	CTestParams& operator=( const CTestParams& other );

	CString GetStrValue( const CString& key ) const;

	template<class T>
	T GetValue( const CString& key ) const;

	template<class T>
	void GetArray( const CString& key, CArray<T>& value ) const;

	CInterval GetInterval( const CString& key ) const;

	friend std::ostream& operator<<( std::ostream& os, const CTestParams& params );

private:
	CMap<CString, CString> flags;

	CInterval parseInterval( const CString& stringValue ) const;
	void splitStringsByDelimiter( CArray<CString>& result, const CString& string, const CString& delimiter ) const;
};

std::ostream& operator<<( std::ostream& os, const CTestParams& params );

//---------------------------------------------------------------------------------------------------------------------

template<class T>
inline T CTestParams::GetValue( const CString& key ) const
{
	T result = T();
	NeoAssert( Value( GetStrValue( key ), result ) );
	return result;
}

template<>
inline void CTestParams::GetArray<CInterval>( const CString& key, CArray<CInterval>& value ) const
{
	NeoAssert( flags.Has( key ) );

	value.DeleteAll();
	CString valueString = flags.Get( key );
	const char* valueStringPtr = valueString;
	const char* leftBracketPtr = strchr( valueStringPtr, '{' );
	const char* rightBracketPtr = strrchr( valueStringPtr, '}' );
	NeoAssert( leftBracketPtr != 0 && rightBracketPtr != 0 );

	CArray<CString> intervalStrings;
	splitStringsByDelimiter( intervalStrings, CString( leftBracketPtr + 1, static_cast<int>( rightBracketPtr - leftBracketPtr - 1 ) ), "," );
	value.SetBufferSize( intervalStrings.Size() );
	for( int i = 0; i < intervalStrings.Size(); ++i ) {
		value.Add( parseInterval( intervalStrings[i] ) );
	}
}

} // namespace NeoMLTest
