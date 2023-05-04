/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/TraditionalML/SubwordEncoderTrainer.h>

namespace NeoML {

class CBpeTrainer {
public:
	CBpeTrainer( int vocabSize, CSubwordEncoderTrainer::TBorderHandling, bool useByteBpe, int unknownTokenId );

	// Trains and returns a fully trained encoder.
	CPtr<IBytePairEncoder> Train( const CWordDictionary& frequencyDict, const CWordDictionary& charVocab );

private:
	// Bpe-vocabulary entry
	struct CToken {
		// Token itself
		CString Text;
		// Flag. <UNK> cannot be merged
		bool IsUnk = false;
	};

	// Just a pair of tokens (bigram)
	struct CCandidatePair {
		// Index in the bpe-vocabulary
		int Left = NotFound;
		int Right = NotFound;

		CCandidatePair() = default;
		CCandidatePair( int left, int right );
		int HashKey() const;
		bool operator ==( const CCandidatePair& other ) const;
		bool operator !=( const CCandidatePair& other ) const { return !( *this == other ); }
	};

	// Extended information about the token-pair
	struct CCandidateData {
		CCandidatePair Pair;
		// Word IDs where this pair was found and a number of copies there.
		CMap<int, int> WordOccurrences;
		// Joint text of a candidate
		CString Text;
		// Total number of occurrences in training corpus.
		int64_t RealCount = 0;
		// This value may be outdated since we don't update values in queue. It is stored for internal comparisons in queue only.
		int64_t QueueCount = 0;
	};

	// Comparator for CCandidateData
	class CCandidateDataComparator {
	public:
		static bool Predicate( const CCandidateData* first, const CCandidateData* second );
		static bool IsEqual( const CCandidateData*, const CCandidateData* ) { return false; }
		void Swap( CCandidateData*& first, CCandidateData*& second ) const { swap( first, second ); }
	};

	// Training dataset entry
	struct CWordWithCount {
		// Indices of bpe-tokens in the vocabulary we construct
		CArray<int> Text;
		int64_t Count = 0;
	};

	// Trainer and encoder params
	int desiredVocabSize;
	CSubwordEncoderTrainer::TBorderHandling borderHandling;
	bool useByteBpe;
	int encoderUnkTokenId = 0;

	// bpe-vocabulary
	CArray<CToken> vocabulary;
	// no need to search for complex tokens. Mapping only for chars
	CMap<const char*, int> charTokenIndex;
	CPointerArray<CString> charTokenStorage;
	// ids of some useful tokens
	int unkToken = NotFound;
	int bowToken = NotFound;
	int eowToken = NotFound;

	// training data
	CArray<CWordWithCount> dataset;

	// pairs of tokens to their extended training information
	// OWNS CCandidateData* used in queue and newCandidates
	CMap<CCandidatePair, CCandidateData> candidates;
	// Max-heap of candidates. 'Ascending' is ok, a queue is MAX-queue by default.
	CPriorityQueue<CArray<CCandidateData*>, CCandidateDataComparator> queue;
	// temp storage for new pairs appeared during the last merge
	CArray<CCandidateData*> newCandidates;

	void addCharToken( const CString& tokenText, bool isUnk );
	
	void prepareDataset( const CWordDictionary& trainData );
	void addAllBigrams();

	void addPair( const CCandidatePair& pair, int wordId, int64_t wordCount );
	CString mergeText( const CCandidatePair& pair ) const;

	void deletePair( const CCandidatePair& pair, int wordId, int64_t wordCount );
	void updateStatistics( const CCandidateData& newTokenData, int newTokenId );
	void updateOneWordStatistics( const CCandidateData& newTokenData, int newTokenId,
		CArray<int>& word, int wordId, int newTokenCountInThisWord );

	void enqueueNewCandidates();

	CPtr<IBytePairEncoder> createEncoder();

	int64_t checkNaive( CCandidateData* );

	friend void ArrayMemMoveElement( CWordWithCount* dest, CWordWithCount* src );
};

inline void ArrayMemMoveElement( CBpeTrainer::CWordWithCount* dest, CBpeTrainer::CWordWithCount* src )
{
	NeoPresume( dest != src );
	::new( dest ) CBpeTrainer::CWordWithCount;
	src->Text.MoveTo( dest->Text );
	dest->Count = src->Count;
	src->~CWordWithCount();
}

} // namespace NeoML
