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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/WordDictionary.h>

namespace NeoML {

// Class that calculates byte-pair-encoding tokens.
class NEOML_API CBpeIterativeBuilder {
public:
	CBpeIterativeBuilder();

	// Initialization from word dictionary and the number of tokens to be calculated.
	void Initialize( const CWordDictionary& dictionary, int totalIterationsCount );

	// Performs the calculation of new BPE tokens.
	CWordDictionary RunIterations( int iterationCount );
	CWordDictionary RunTotalIterations();

	// Returns the number of iterations completed (== number of calculated token).
	int IterationsCompletedCount() const { return iterationsCompletedCount; }
	// Returns the total number of iterations.
	int TotalIterationsCount() const { return totalIterationsCount; }
	// Returns true if no more iterations can be performed.
	bool IsBuildCompleted() const;

	// Serialization to archive.
	void Serialize( CArchive& archive );

private:
	// The total number of iterations.
	// The size of the dictionary returned from RunTotalIterations cannot exceed this value.
	int totalIterationsCount;
	// The number of completed iterations.
	int iterationsCompletedCount;

	// The current state of train word dictionary.
	CWordDictionary trainDictionary;
	// The dictionary of pairs of neighbour tokens.
	CWordDictionary pairDictionary;

	// Map: pair of neighbour tokens -> set of ids of words containing this pair of tokens.
	typedef CMap<CString, CHashTable<int>> CPairReverseIndex;
	CPairReverseIndex reverseIndex;

	int calcIterationsCount( int requestedIterationsCount ) const;
	void buildPairDictionary( CWordDictionary& newTokens );
	bool runSingleIteration( CWordDictionary& newTokens );
};

// Class that encodes a word using byte-pair-encoding.
class NEOML_API CBytePairEncoder {
public:
	CBytePairEncoder() = default;
	CBytePairEncoder( const CBytePairEncoder& other );
	CBytePairEncoder& operator=( const CBytePairEncoder& other );

	// Builds encoder.
	void Build( const CWordDictionary& vocabulary, int size );
	// Returns tokens dictionary.
	const CWordDictionary& GetTokens() const { return tokens; }

	// Initializes a builder that contains all the neccessary data for bpe tokens calculation
	// and supports serialization.
	// Thus, BPE tokens calculations can be performed asyncronically.
	void InitializeBuilder( const CWordDictionary& vocabulary, int size,
		CBpeIterativeBuilder& builder );
	// Updates tokens with new ones (usually gotten from CBpeIterativeBuilder).
	void UpdateTokens( const CWordDictionary& newTokens );

	// Encodes a word.
	void Encode( const CString& word, CArray<int>& tokenIds, CArray<int>& offsets ) const;
	// Returns the number of tokens.
	int Size() const { return tokens.Size(); }
	
	// Serialization to archive.
	void Serialize( CArchive& archive );

	// Sets the cache cleanup period.
	// The cache is used for Encode calls acceleration.
	// The result of the encode call is cached and will be erased if 
	// no call with the same word will occur among next 1-2 X cachePeriod calls.
	// Increase in cachePeriod leads to a in increase in memory consumption.
	// To completely switch the cache off set cachePeriod equal to -1.
	// Value 0 is treated as invalid.
	void SetCachePeriod( int cachePeriod ) const { cache.SetCachePeriod( cachePeriod ); }

private:
	// BPE tokens.
	CWordDictionary tokens;

	// Internal cache for frequent encoding requests.
	class CCache {
	public:
		CCache();
		// Sets the cache cleanup period
		void SetCachePeriod( int newPeriod );
		// Requests data from cache.
		bool Request( const CString& word, CArray<int>& bpeEncoding );
		// Adds data to cache.
		void Add( const CString& word, const CArray<int>& bpeEncoding );

	private:
		// Data stored in cache: encoding and the lattest request time.
		struct CEncodedWord {
			CFastArray<int, 4> TokenIds;
			long long Time;

			CEncodedWord() : Time( 0 ) {}
			CEncodedWord( const CEncodedWord& other ) :
				Time( other.Time ) {
				other.TokenIds.CopyTo( TokenIds );
			}
		};

		// Current cache state.
		CMap<CString, CEncodedWord> wordCache;
		// Current cache time.
		long long cacheTime;
		// Cache cleanup period.
		int cachePeriod;
	};

	// Cache for Encode calls.
	mutable CCache cache;

	static const int currentVersion = 0;

	void createTrainVocabulary( const CWordDictionary& vocabulary,
		CWordDictionary& trainDictionary ) const;
	CString splitWordIntoInitalTokens( const CString& word ) const;
	//CString removeSpecialTokens( const CString& word ) const;
	//void calculateOffsets( const CArray<int>& tokenIds,
	//	CArray<int>& offsets ) const;
};

} // namespace NeoML
