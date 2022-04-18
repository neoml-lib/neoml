/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "PyBytePairEncoder.h"
#include "PyMemoryFile.h"

class CPyBytePairEncoder {
public:
	CPyBytePairEncoder();
	
	void Train( py::dict vocabulary, int tokensCount, bool useEndOfWordToken,
		bool useStartOfWordToken );
	py::tuple Encode( const std::string& word ) const;
	py::list Decode( py::list tokenIds ) const;

	const CBytePairEncoder& Encoder() const { return *encoder; }
	CBytePairEncoder& Encoder() { return *encoder; }

private:
	std::unique_ptr<CBytePairEncoder> encoder;
};

CPyBytePairEncoder::CPyBytePairEncoder() :
	encoder( new CBytePairEncoder() )
{}

void CPyBytePairEncoder::Train( py::dict vocabulary, int tokensCount,
	bool useEndOfWordToken, bool useStartOfWordToken )
{
	CWordDictionary vocabularyRaw;
	for( const auto& item : vocabulary ) {
		vocabularyRaw.AddWord( item.first.cast<std::string>(), 
			item.second.cast<int>() );
	}
	{
		py::gil_scoped_release release;
		encoder->Train( vocabularyRaw, tokensCount, useEndOfWordToken, useStartOfWordToken );
	}
}

py::tuple CPyBytePairEncoder::Encode( const std::string& word ) const
{
	CArray<int> tokenIds;
	CArray<int> tokenLengths;
	encoder->Encode( word, tokenIds, tokenLengths );

	py::list tokenIdsResult;
	py::list tokenLengthsResult;

	int shift = 0;
	for( int i = 0; i < tokenIds.Size(); i++ ) {
		tokenIdsResult.append( tokenIds[i] );
		tokenLengthsResult.append( py::make_tuple( shift, shift + tokenLengths[i] ) );
		shift += tokenLengths[i];
	}

	return py::make_tuple( tokenIdsResult, tokenLengthsResult );
}

py::list CPyBytePairEncoder::Decode( py::list _tokenIds ) const
{
	CArray<int> tokenIds;
	for( const auto& id : _tokenIds ) {
		tokenIds.Add( id.cast<int>() );
	}

	CArray<CString> words;
	encoder->Decode( tokenIds, words );

	py::list resultWords;
	for( int i = 0; i < words.Size(); i++ ) {
		resultWords.append( std::string( words[i] ) );
	}

	return resultWords;
}

void InitializeBytePairEncoder( py::module& m )
{
	py::class_<CPyBytePairEncoder>(m, "BytePairEncoder")
		.def( py::init<>() )
		.def( "train", &CPyBytePairEncoder::Train, py::return_value_policy::reference )
		.def( "encode", &CPyBytePairEncoder::Encode, py::return_value_policy::reference )
		.def( "decode", &CPyBytePairEncoder::Decode, py::return_value_policy::reference )
		.def( py::pickle(
			[]( const CPyBytePairEncoder& pyBpe ) {
				CPyMemoryFile file;
				CArchive archive( &file, CArchive::store );
				const_cast<CBytePairEncoder&>( pyBpe.Encoder() ).Serialize( archive );
				archive.Close();
				file.Close();
				return py::make_tuple( file.GetBuffer() );
			},
			[]( py::tuple t ) {
				if( t.size() != 1 ) {
					throw std::runtime_error( "Invalid state!" );
				}

				auto t0_array = t[0].cast<py::array>();
				CPyMemoryFile file( t0_array );
				CArchive archive( &file, CArchive::load );
				CPyBytePairEncoder pyBpe;
				pyBpe.Encoder().Serialize( archive );
				return pyBpe;
			}
			) 
		)
	;
}
