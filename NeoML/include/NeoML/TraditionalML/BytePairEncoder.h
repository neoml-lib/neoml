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
class NEOML_API CBpeIterativeTrainer {
public:
	CBpeIterativeTrainer();
	CBpeIterativeTrainer( const CBpeIterativeTrainer& other ) = delete;
	CBpeIterativeTrainer& operator=( const CBpeIterativeTrainer& other ) = delete;

	void Initialize( const CArray<CArray<CString>>& trainSplittedWords, 
		const CArray<long long>& trainWordCounts, int totalIterationsCount );

	// Performs the calculation of new BPE tokens.
	CWordDictionary RunIterations( int iterationCount );
	CWordDictionary RunTotalIterations();

	// Returns the number of iterations completed (== number of calculated token).
	int IterationsCompletedCount() const { return iterationsCompletedCount; }
	// Returns the total number of iterations.
	int TotalIterationsCount() const { return totalIterationsCount; }
	// Returns true if no more iterations can be performed.
	bool IsCompleted() const;

	// Serialization to archive.
	void Serialize( CArchive& archive );

private:
	// The total number of iterations.
	// The size of the dictionary returned from RunTotalIterations cannot exceed this value.
	int totalIterationsCount;
	// The number of completed iterations.
	int iterationsCompletedCount;

	// The dictionary of pairs of neighbour tokens.
	CWordDictionary pairDictionary;

	CArray<CArray<CString>> trainWords;
	CArray<long long> trainCounts;

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
	CBytePairEncoder();
	CBytePairEncoder( const CBytePairEncoder& other );
	CBytePairEncoder& operator=( const CBytePairEncoder& other );

	// Trains encoder.
	void Train( const CWordDictionary& dictionary, int size,
		bool useEndOfWordToken = true, bool useStartOfWordToken = false );
	// Returns tokens dictionary.
	const CWordDictionary& GetTokens() const { return tokens; }

	// Initializes a trainer that contains all the neccessary data for bpe tokens training
	// and supports serialization.
	// Thus, BPE tokens training can be performed asyncronically.
	void InitializeTrainer( const CWordDictionary& vocabulary, int size,
		CBpeIterativeTrainer& builder );
	// Updates tokens with new ones (usually gotten from CBpeIterativeTrainer).
	void UpdateTokens( const CWordDictionary& newTokens );

	// Encodes a word.
	// Input word should be encoded in UTF-8.
	// TokenId range = [-1, 0, ... , Size() - 1] where -1 is reserved for unknown characters.
	// Some special tokens (like End-Of-Word) have zero unicode length.
	void Encode( const CString& word, CArray<int>& tokenIds, 
		CArray<int>& unicodeTokenLengths ) const;
	// Decodes sequence of token ids into sequnce of words.
	// tokenId == -1 (unknown) is decoded as <UNK>.
	void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const;
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
	// Special flags usage flags.
	bool useEndOfWordToken;
	bool useStartOfWordToken;

	// Internal cache for frequent encoding requests.
	class CCache {
	public:
		CCache();
		// Sets the cache cleanup period
		void SetCachePeriod( int newPeriod );
		// Requests data from cache.
		bool Request( const CString& word, CArray<int>& tokenIds,
			CArray<int>& tokenLengths );
		// Adds data to cache.
		void Add( const CString& word, const CArray<int>& tokenIds,
			const CArray<int>& tokenLengths );

	private:
		// Data stored in cache: token ids and their uniode lengths and the lattest request time.
		struct CEncodedWord {
			CFastArray<int, 4> TokenIds;
			CFastArray<int, 4> TokenLengths;
			long long Time;

			CEncodedWord() : Time( 0 ) {}
			CEncodedWord( const CEncodedWord& other ) :
				Time( other.Time )
			{
				other.TokenIds.CopyTo( TokenIds );
				other.TokenLengths.CopyTo( TokenLengths );
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

	void createTrainData( const CWordDictionary& dictionary,
		CArray<CArray<CString>>& trainWords, CArray<long long>& trainCounts ) const;
	void splitWordIntoInitalTokens( const CString& word, 
		CArray<CString>& splittedWord, CArray<int>* initialLengths = nullptr ) const;
	void removeSpecialTokens( CString& token, bool& hasEoW, bool& hasSoW ) const;
	bool removeEoWToken( CString& token ) const;
	bool removeSoWToken( CString& token ) const;
};

} // namespace NeoML
