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

#pragma once

#include <NeoML/NeoMLDefs.h>

namespace NeoML {

// Dictionary of words with counts and frequencies.
// Words are stored in UTF-8.
class NEOML_API CWordDictionary {
public:
	CWordDictionary() : totalWordsUseCount( 0 ) {}	
	CWordDictionary( CWordDictionary&& other );
	CWordDictionary& operator=( CWordDictionary&& other );

	// To deny implicit expensive copy calls.
	CWordDictionary( const CWordDictionary& other ) = delete;
	CWordDictionary& operator=( const CWordDictionary& other ) = delete;

	// Replaces data in 'other' with data from 'this'.
	void CopyTo( CWordDictionary& other ) const;
	
	// Adds word to the dictionary and increases its count by the given value.
	void AddWord( const CString& word, long long count = 1 );
	// Add all words from newDictionary to the dictionary.
	void AddDictionary( const CWordDictionary& newDictionary );
	// Finishes the dictionary build process.
	// Removes all words with a total count less than minUseCount.
	// Sorts all remaining words by their counts and assigns id to each word equal to their position number.
	// Thus change to the dictionary caused by series of AddWord/AddDictionary calls 
	// should be followed by Finalize call.
	void Finalize( long long minUseCount );

	// Returns word id.
	// If word is absent returns -1.
	int GetWordId( const CString& word ) const;
	// Checks if the dictionary contains given word or not.
	bool HasWord( const CString& word ) const;
	
	// Returns word by id.
	// id must be valid.
	CString GetWord( int id ) const;

	// Returns accumulated value of counts from all AddWord calls for the given word.
	long long GetWordUseCount( const CString& word ) const;
	long long GetWordUseCount( int id ) const;
	// Returns frequency of the given word.
	double GetWordFrequency( const CString& word ) const;
	double GetWordFrequency( int id ) const;

	// Returns number of words in the dictionary.
	int Size() const { return words.Size(); }
	// Checks whether dictionary is empty or not.
	bool IsEmpty() const { return Size() == 0; }
	// Decreases size of the dictionary if maxSize is less then the current size.
	void RestrictSize( int maxSize );
	// Removes all the words from the dictionary.
	void Empty();

	// Dictionary serialization.
	void Serialize( CArchive& archive );

private:
	// Word with accumulated count.
	struct CWordWithCount {
		// Word.
		CString Word;
		// Accumulated count from all AddWord calls.
		long long Count;

		CWordWithCount() : Count( 0 ) {}
		CWordWithCount( const CString& word, long long count ) : Word( word ), Count( count ) {}

		void Serialize( CArchive& archive );

		bool operator<( const CWordWithCount& other ) const;
		bool operator==( const CWordWithCount& other ) const;
	};

	// Words sorted by their count.
	CArray<CWordWithCount> words;
	// Map: word -> position in word array.
	CMap<CString, int> wordToId;
	// Sum of count of all words.
	long long totalWordsUseCount;

	void buildIndex();
	void checkId( int wordId ) const;

	friend CArchive& operator<<( CArchive& archive, const CWordWithCount& value );
	friend CArchive& operator>>( CArchive& archive, CWordWithCount& value );
};

inline CArchive& operator<<( CArchive& archive, const CWordDictionary::CWordWithCount& value )
{
	const_cast<CWordDictionary::CWordWithCount&>( value ).Serialize( archive );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, CWordDictionary::CWordWithCount& value )
{
	value.Serialize( archive );
	return archive;
}

} // namespace NeoML
