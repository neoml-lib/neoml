/* Copyright © 2017-2021 ABBYY Production LLC

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
	CPyBytePairEncoder() = default;
	
	void Build( py::dict vocabulary, int tokensCount );
	py::list Encode( const std::string& word ) const;
	std::string Decode( py::list encoding ) const;

	const CBytePairEncoder& Encoder() const { return encoder; }
	CBytePairEncoder& Encoder() { return encoder; }

private:
	CBytePairEncoder encoder;
};

void CPyBytePairEncoder::Build( py::dict vocabulary, int tokensCount )
{
	CWordVocabulary vocabularyRaw;
	for( const auto& item : vocabulary ) {
		vocabularyRaw.AddWordWithCount( item.first.cast<std::string>(), 
			item.second.cast<int>() );
	}
	{
		py::gil_scoped_release release;
		encoder.Build( tokensCount, vocabularyRaw );
	}
}

py::list CPyBytePairEncoder::Encode( const std::string& word ) const
{
	CArray<int> encoding;
	encoder.Encode( word, encoding );

	py::list result;
	for( int i = 0; i < encoding.Size(); i++ ) {
		result.append( encoding[i] );
	}
	return result;
}

std::string CPyBytePairEncoder::Decode( py::list encoding ) const
{
	CArray<int> encodingRaw;
	encodingRaw.SetSize( encoding.size() );
	for( int i = 0; i < encoding.size(); i++ ) {
		encodingRaw[i] = encoding[i].cast<int>();
	}
	return encoder.Decode( encodingRaw );
}

void InitializeBytePairEncoder( py::module& m )
{
	py::class_<CPyBytePairEncoder>(m, "BytePairEncoder")
		.def( py::init<>() )
		.def( "build", &CPyBytePairEncoder::Build, py::return_value_policy::reference )
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
