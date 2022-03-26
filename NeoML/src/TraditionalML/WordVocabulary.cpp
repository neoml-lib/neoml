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

#include <NeoML/TraditionalML/WordVocabulary.h>

namespace NeoML {

bool CWordVocabulary::CWordWithCount::operator<( const CWordVocabulary::CWordWithCount& other ) const
{
	if( Count < other.Count ) {
		return true;
	} else if( Count == other.Count ) {
		return Word > other.Word;
	} else {
		return false;
	}
}

bool CWordVocabulary::CWordWithCount::operator==( const CWordVocabulary::CWordWithCount& other ) const
{
	return Count == other.Count
		&& Word == other.Word;
}

void CWordVocabulary::CWordWithCount::Serialize( CArchive& archive )
{
	archive.Serialize( Word );
	archive.Serialize( Count );
}

//////////////////////////////////////////////////////////////////////////////////////////

CWordVocabulary::CWordVocabulary( const CWordVocabulary& other ) :
	totalWordsCount( other.totalWordsCount )
{
	other.wordToIndex.CopyTo( wordToIndex );
	other.words.CopyTo( words );
}

CWordVocabulary& CWordVocabulary::operator=( const CWordVocabulary& other )
{
	other.wordToIndex.CopyTo( wordToIndex );
	other.words.CopyTo( words );
	return *this;
}

#pragma message (WARNING_PREFIX "move support")

//////////////////////////////////////////////////////////////////////////////////////////

int CWordVocabulary::GetWordIndex( const CString& word ) const
{
	int index = NotFound;
	if( !wordToIndex.Lookup( word, index ) ) {
		return NotFound;
	}
	return index;
}

bool CWordVocabulary::HasWord( const CString& word ) const
{
	return GetWordIndex( word ) != NotFound;
}

CString CWordVocabulary::GetWord( int index ) const
{
	checkIndex( index );
	return words[index].Word;
}

void CWordVocabulary::SetWord( int index, const CString& newWord )
{
	checkIndex( index );
	const CString oldWord = GetWord( index );
	wordToIndex.Delete( oldWord );
	wordToIndex.Set( newWord, index );
	words[index].Word = newWord;
}

void CWordVocabulary::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );
	if( archive.IsStoring() ) {
		NeoAssert( words.IsSorted<Descending<CWordWithCount>>() );
	}
	words.Serialize( archive );
	if( archive.IsLoading() ) {
		buildIndex();
	}
	archive.Serialize( totalWordsCount );
}

void CWordVocabulary::RestrictSize( int maxSize )
{
	if( maxSize < Size() ) {
		words.SetSize( maxSize );
	}
}

void CWordVocabulary::buildIndex()
{
	wordToIndex.Empty();
	wordToIndex.SetHashTableSize( words.Size() );
	for( int i = 0; i < words.Size(); i++ ) {
		wordToIndex.Set( words[i].Word, i );
	}
}

long long CWordVocabulary::GetWordUseCount( const CString& word ) const
{
	const int index = GetWordIndex( word );
	if( index == NotFound ) {
		return 0;
	}
	return GetWordUseCount( index );
}

double CWordVocabulary::GetWordFrequency( int index ) const
{
	checkIndex( index );
	NeoAssert( totalWordsCount > 0 );
	return static_cast<double>( words[index].Count ) / totalWordsCount;
}

double CWordVocabulary::GetWordFrequency( const CString& word ) const
{
	const int index = GetWordIndex( word );
	if( index == NotFound ) {
		return 0.0;
	}
	return GetWordFrequency( index );
}

void CWordVocabulary::AddWord( const CString& word )
{
	AddWordWithCount( word, 1 );
}

void CWordVocabulary::AddWordWithCount( const CString& word, long long addedCount )
{
	int index = 0;
	if( !wordToIndex.Lookup( word, index ) ) {
		wordToIndex.Set( word, words.Size() );
		words.Add( CWordWithCount{ word, addedCount } );
	} else {
		words[index].Count += addedCount;
	}
	totalWordsCount += addedCount;
}

void CWordVocabulary::AddVocabulary( const CWordVocabulary& vocabulary )
{
	for( int i = 0; i < vocabulary.Size(); i++ ) {
		const CString word = vocabulary.GetWord( i );
		const long long count = vocabulary.GetWordUseCount( i );
		AddWordWithCount( word, count );
	}
}

void CWordVocabulary::Finalize( long long minCount )
{
	words.QuickSort<Descending<CWordWithCount>>();

	const CWordWithCount countSeparator{ "", minCount - 1 };
	const int insertionPoint = words.FindInsertionPoint<Descending<CWordWithCount>>( countSeparator );
	
#pragma message( WARNING_PREFIX "Need test")
	RestrictSize( insertionPoint + 1 );
	buildIndex();
}

void CWordVocabulary::Empty()
{
	totalWordsCount = 0;
	wordToIndex.Empty();
	words.Empty();
}

void CWordVocabulary::checkIndex( int index ) const
{
	NeoAssert( index >= 0 && index < Size() );
}

} // namespace NeoML