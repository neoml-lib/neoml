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

#include <NeoML/TraditionalML/WordDictionary.h>

namespace NeoML {

bool CWordDictionary::CWordWithCount::operator<( const CWordDictionary::CWordWithCount& other ) const
{
	if( Count < other.Count ) {
		return true;
	} else if( Count == other.Count ) {
		return Word > other.Word;
	} else {
		return false;
	}
}

bool CWordDictionary::CWordWithCount::operator==( const CWordDictionary::CWordWithCount& other ) const
{
	return Count == other.Count
		&& Word == other.Word;
}

void CWordDictionary::CWordWithCount::Serialize( CArchive& archive )
{
	archive.Serialize( Word );
	archive.Serialize( Count );
}

//////////////////////////////////////////////////////////////////////////////////////////

CWordDictionary::CWordDictionary( CWordDictionary&& other ) :
	totalWordsUseCount( other.totalWordsUseCount )
{
	other.words.MoveTo( words );
	other.wordToId.MoveTo( wordToId );
}

CWordDictionary& CWordDictionary::operator=( CWordDictionary&& other )
{
	totalWordsUseCount = other.totalWordsUseCount;
	other.words.MoveTo( words );
	other.wordToId.MoveTo( wordToId );
	return *this;
}

void CWordDictionary::CopyTo( CWordDictionary& other ) const
{
	other.totalWordsUseCount = totalWordsUseCount;
	words.CopyTo( other.words );
	wordToId.CopyTo( other.wordToId );
}

void CWordDictionary::AddWord( const CString& word, long long count )
{
	int id = 0;
	if( !wordToId.Lookup( word, id ) ) {
		NeoAssert( count > 0 );
		wordToId.Set( word, words.Size() );
		words.Add( CWordWithCount{ word, count } );
	} else {
		words[id].Count += count;
		NeoAssert( words[id].Count >= 0 );
	}
	totalWordsUseCount += count;
	NeoAssert( totalWordsUseCount >= 0 );
}

void CWordDictionary::AddDictionary( const CWordDictionary& newDictionary )
{
	for( int i = 0; i < newDictionary.Size(); i++ ) {
		const CString word = newDictionary.GetWord( i );
		const long long count = newDictionary.GetWordUseCount( i );
		AddWord( word, count );
	}
}

void CWordDictionary::Finalize( long long minCount )
{
	words.QuickSort<Descending<CWordWithCount>>();

	if( minCount > INT64_MIN ) {
		const CWordWithCount countSeparator{ "", minCount - 1 };
		const int insertionPoint = words.FindInsertionPoint<Descending<CWordWithCount>>( countSeparator );
		RestrictSize( insertionPoint );
	}

	buildIndex();
}

int CWordDictionary::GetWordId( const CString& word ) const
{
	int id = NotFound;
	if( !wordToId.Lookup( word, id ) ) {
		return NotFound;
	}
	return id;
}

bool CWordDictionary::HasWord( const CString& word ) const
{
	return GetWordId( word ) != NotFound;
}

CString CWordDictionary::GetWord( int id ) const
{
	checkId( id );
	return words[id].Word;
}

long long CWordDictionary::GetWordUseCount( int id ) const
{
	checkId( id );
	return words[id].Count;
}

long long CWordDictionary::GetWordUseCount( const CString& word ) const
{
	const int id = GetWordId( word );
	if( id == NotFound ) {
		return 0;
	}
	return GetWordUseCount( id );
}

double CWordDictionary::GetWordFrequency( int id ) const
{
	checkId( id );
	NeoAssert( totalWordsUseCount > 0 );
	return static_cast<double>( words[id].Count ) / totalWordsUseCount;
}

double CWordDictionary::GetWordFrequency( const CString& word ) const
{
	const int id = GetWordId( word );
	if( id == NotFound ) {
		return 0.0;
	}
	return GetWordFrequency( id );
}

void CWordDictionary::RestrictSize( int maxSize )
{
	if( maxSize < Size() ) {
		words.SetSize( maxSize );
	}
}

void CWordDictionary::Empty()
{
	totalWordsUseCount = 0;
	wordToId.Empty();
	words.Empty();
}

void CWordDictionary::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );
	words.Serialize( archive );
	if( archive.IsLoading() ) {
		buildIndex();
	}
	archive.Serialize( totalWordsUseCount );
}

// Builds reverse index: word -> word id.
void CWordDictionary::buildIndex()
{
	wordToId.Empty();
	wordToId.SetHashTableSize( words.Size() );
	for( int i = 0; i < words.Size(); i++ ) {
		wordToId.Set( words[i].Word, i );
	}
}

// Checks word id validity.
void CWordDictionary::checkId( int id ) const
{
	NeoAssert( id >= 0 && id < Size() );
}

} // namespace NeoML
