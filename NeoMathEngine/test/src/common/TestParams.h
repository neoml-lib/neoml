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

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

namespace NeoMLTest {

bool Value( const std::string& str, int& result );
bool Value( const std::string& str, double& result );

struct CInterval {
	int Begin;
	int End;
};

class CTestParams {
public:
	explicit CTestParams( const std::string& str );
	CTestParams( const CTestParams& other );

	CTestParams& operator=( const CTestParams& other );

	std::string GetStrValue( const std::string& key ) const;

	template<class T>
	T GetValue( const std::string& key ) const;
	
	template<class T>
	void GetArray( const std::string& key, std::vector<T>& value ) const;

	CInterval GetInterval( const std::string& key ) const;

	friend ::std::ostream& operator<<( ::std::ostream& os, const CTestParams& params );

private:
	std::unordered_map<std::string, std::string> flags;

	CInterval parseInterval( const std::string& stringValue ) const;
	void splitStringsByDelimiter( std::vector<std::string>& result, const std::string& string, const std::string& delimiter ) const;
};

::std::ostream& operator<<( ::std::ostream& os, const CTestParams& params );

//---------------------------------------------------------------------------------------------------------------------

template<class T>
inline T CTestParams::GetValue( const std::string& key ) const
{
	T result = T();
	Value( GetStrValue( key ), result );
	return result;
}

template<>
inline void CTestParams::GetArray<CInterval>( const std::string& key, std::vector<CInterval>& value ) const
{
	value.clear();
	std::string valueString = flags.find( key )->second;
	const char* valueStringPtr = valueString.c_str();
	const char* leftBracketPtr = strchr( valueStringPtr, '{' );
	const char* rightBracketPtr = strrchr( valueStringPtr, '}' );

	std::vector<std::string> intervalStrings;
	splitStringsByDelimiter( intervalStrings, std::string( leftBracketPtr + 1, static_cast<int>( rightBracketPtr - leftBracketPtr - 1 ) ), "," );
	value.reserve( intervalStrings.size() );
	for( size_t i = 0; i < intervalStrings.size(); ++i ) {
		value.push_back( parseInterval( intervalStrings[i] ) );
	}
}

} // namespace NeoMLTest
