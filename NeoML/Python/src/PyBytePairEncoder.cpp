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
	
	void Train( py::dict dictionary, int maxVocabSize, bool useEndOfWordToken,
		bool useStartOfWordToken );

	void LoadFromDictionary( py::dict dictionary, 
		const std::string& endOfWordToken, const std::string& startOfWordToken );
	py::dict GetDictionary() const;

	void Load( const std::string& path );
	void Store( const std::string& path );

	py::tuple Encode( py::list text ) const;
	py::list Decode( py::list tokenIds ) const;

	int Size() const;
	bool UseEoW() const;
	bool UseSoW() const;

	void SetCachePeriod( int period ) const;
	int GetCachePeriod() const;

	void Serialize( CArchive& archive );

private:
	CPtr<IBytePairEncoder> encoder;
};

CPyBytePairEncoder::CPyBytePairEncoder() :
	encoder( nullptr )
{}

void CPyBytePairEncoder::Train( py::dict dictionary, int maxVocabSize,
	bool useEndOfWordToken, bool useStartOfWordToken )
{
	CWordDictionary dictionaryRaw;
	for( const auto& item : dictionary ) {
		dictionaryRaw.AddWord( item.first.cast<std::string>(),
			item.second.cast<int>() );
	}
	{
		py::gil_scoped_release release;

		CBytePairEncoderTrainer::CParams params;
		params.MaxSize = maxVocabSize;
		params.UseEndOfWordToken = useEndOfWordToken;
		params.UseStartOfWordToken = useStartOfWordToken;

		CBytePairEncoderTrainer trainer( params, dictionaryRaw );
		encoder = trainer.Train();
	}
}

py::dict CPyBytePairEncoder::GetDictionary() const
{
	NeoAssert( encoder != nullptr );

	py::dict result;
	CWordDictionary dictionary;
	{
		py::gil_scoped_release release;
		encoder->GetDictionary( dictionary );
	}
	for( int i = 0; i < dictionary.Size(); ++i ) {
		result[dictionary.GetWord( i )] = dictionary.GetWordUseCount( i );
	}
	return result;
}

void CPyBytePairEncoder::LoadFromDictionary( py::dict dictionary, 
	const std::string& useEndOfWordToken, const std::string& useStartOfWordToken )
{
	CWordDictionary dictionaryRaw;
	for( const auto& item : dictionary ) {
		dictionaryRaw.AddWord( item.first.cast<std::string>(), item.second.cast<int>() );
	}
	encoder = CheckCast<IBytePairEncoder>( CreateModel( BytePairEncoderModelName ) );
	encoder->LoadDictionary( dictionaryRaw, useEndOfWordToken, useStartOfWordToken );
}

void CPyBytePairEncoder::Load( const std::string& path )
{
	py::gil_scoped_release release;
	CArchiveFile file( path.c_str(), CArchive::load );
	CArchive archive( &file, CArchive::load );
	Serialize( archive );
}

void CPyBytePairEncoder::Store( const std::string& path )
{
	py::gil_scoped_release release;
	CArchiveFile file( path.c_str(), CArchive::store );
	CArchive archive( &file, CArchive::store );
	Serialize( archive );
}

py::tuple CPyBytePairEncoder::Encode( py::list text ) const
{
	NeoAssert( encoder != nullptr );

	CArray<int> tokenIds;
	CArray<int> tokenLengths;

	for( const auto& word : text ) {
		auto cWord = word.cast<std::string>();
		encoder->Encode( cWord, tokenIds, tokenLengths );
	}

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
	NeoAssert( encoder != nullptr );

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

int CPyBytePairEncoder::Size() const
{
	NeoAssert( encoder != nullptr );
	return encoder->Size();
}

bool CPyBytePairEncoder::UseEoW() const 
{
	NeoAssert( encoder != nullptr );
	return encoder->UseEndOfWordToken(); 
}

bool CPyBytePairEncoder::UseSoW() const 
{
	NeoAssert( encoder != nullptr );
	return encoder->UseStartOfWordToken(); 
}

void CPyBytePairEncoder::SetCachePeriod( int period ) const
{
	NeoAssert( encoder != nullptr );
	return encoder->SetCachePeriod( period );
}

int CPyBytePairEncoder::GetCachePeriod() const
{
	NeoAssert( encoder != nullptr );
	return encoder->GetCachePeriod();
}

void CPyBytePairEncoder::Serialize( CArchive& archive )
{
	SerializeModel( archive, encoder );
}

void InitializeBytePairEncoder( py::module& m )
{
	py::class_<CPyBytePairEncoder>(m, "BytePairEncoder")
		.def( py::init<>() )
		.def( "train", &CPyBytePairEncoder::Train )
		.def( "load_from_dictionary", &CPyBytePairEncoder::LoadFromDictionary, py::return_value_policy::reference )
		.def( "load",  &CPyBytePairEncoder::Load )
		.def( "store",  &CPyBytePairEncoder::Store )
		.def( "encode", &CPyBytePairEncoder::Encode )
		.def( "decode", &CPyBytePairEncoder::Decode )
		.def( "get_size", &CPyBytePairEncoder::Size )
		.def( "get_dictionary", &CPyBytePairEncoder::GetDictionary )
		.def( "use_eow", &CPyBytePairEncoder::UseEoW )
		.def( "use_sow", &CPyBytePairEncoder::UseSoW )
		.def( "set_cache_period", &CPyBytePairEncoder::SetCachePeriod )
		.def( "get_cache_period", &CPyBytePairEncoder::GetCachePeriod )
		.def( py::pickle(
			[]( const CPyBytePairEncoder& pyBpe ) {
				CPyMemoryFile file;
				CArchive archive( &file, CArchive::store );
				const_cast<CPyBytePairEncoder&>( pyBpe ).Serialize( archive );
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
				pyBpe.Serialize( archive );
				return pyBpe;
			}
			) 
		)
	;
}
