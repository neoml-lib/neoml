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

#include <NeoML/TraditionalML/WordDictionary.h>
#include <NeoML/TraditionalML/SubwordEncoder.h>

namespace NeoML {
class CBytePairEncoder;

// Class that trains byte-pair-encoding.
class NEOML_API CBytePairEncoderTrainer {
public:
	enum class TBorderHandling {
		// Add special EndOfWord symbol (</s>) to all words
		EndOfWord,
		// Add special BeginOfWord symbol <s> to all words
		BeginOfWord,
		// Same as BeginOfWord, but with U+2581 as <s>.
		// Note that our encoder has no special mode to handle this option. It is a user responsibility to place U+2581.
		SentencePiece,
		// No preprocessing
		None
	};

	enum class TVocabPruning {
		// Restrict a single-letter vocabulary based on their frequency. Default coverage is 1, all symbols will be included into the vocabulary.
		Coverage,
		// Treat training data as raw bytes. Initial vocabulary size is 255, no <UNK> symbols will appear.
		ByteBPE
	};

	CBytePairEncoderTrainer( int vocabSize, TBorderHandling, TVocabPruning = TVocabPruning::Coverage );
	~CBytePairEncoderTrainer();

	// Prune single-letter vocabulary so that it covers 'fraction' of the training data. Useful when text contains many rare unicode symbols.
	// By default initial vocabulary contains all found chars (fraction = 1)
	void SetCharacterCoverage( double fraction );
	// Explicitly define required letters that cannot be deleted while pruning
	void AddMandatoryChars( const CArray<CString>& );
	// 0 by default. All other tokens will have contiguous numbers from ( UnknownTokenId + 1 )
	void SetUnknownTokenId( int value );

	// Trains and returns a fully trained encoder.
	CPtr<IBytePairEncoder> Train( const CWordDictionary& frequencyDict );

	CBytePairEncoderTrainer( const CBytePairEncoderTrainer& other ) = delete;
	CBytePairEncoderTrainer& operator=( const CBytePairEncoderTrainer& other ) = delete;

private:
	struct CToken {
		// Token itself
		CString Text;
		// Flag. <UNK> cannot be merged
		bool IsUnk = false;
	};

	struct CWordWithCount {
		CArray<int> Text;
		int64_t Count = 0;
	};

	// Just a bigram
	struct CCandidatePair {
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
		CCandidateDataComparator( bool inQueue );
		bool Predicate( const CCandidateData* first, const CCandidateData* second ) const;
		bool IsEqual( const CCandidateData*, const CCandidateData* ) const { return false; }
		void Swap( CCandidateData*& first, CCandidateData*& second ) const { swap( first, second ); }
	private:
		// In queue: use different member, reverse predicate
		bool inQueue;
	};

	// Trainer and encoder params
	int desiredVocabSize;
	TBorderHandling borderHandling;
	TVocabPruning vocabPruning;
	double coverage = 1.;
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

	void addCharToken( CToken&& token );
	CWordDictionary getInitialDictionary( const CWordDictionary& trainData ) const;
	static CWordDictionary getAllBytesDictionary();

	void prepareDataset( const CWordDictionary& trainData );
	void addAllBigrams();

	void addPair( const CCandidatePair& pair, int wordId, int64_t wordCount );
	CString mergeText( const CCandidatePair& pair ) const;

	void deletePair( const CCandidatePair& pair, int wordId, int64_t wordCount );
	void updateStatistics( const CCandidateData& newTokenData, int newTokenId );
	void updateOneWordStatistics( const CCandidateData& newTokenData, int newTokenId,
		CArray<int>& word, int wordId, int newTokenCountInThisWord );

	void enqueueNewCandidates();
	void dropCandidates( int desiredSize );

	CPtr<IBytePairEncoder> createEncoder();

	int64_t checkNaive( CCandidateData* );

	friend void ArrayMemMoveElement( CWordWithCount* dest, CWordWithCount* src );
};

inline void ArrayMemMoveElement( CBytePairEncoderTrainer::CWordWithCount* dest, CBytePairEncoderTrainer::CWordWithCount* src )
{
	NeoPresume( dest != src );
	::new( dest ) CBytePairEncoderTrainer::CWordWithCount;
	src->Text.MoveTo( dest->Text );
	dest->Count = src->Count;
	src->~CWordWithCount();
}

} // namespace NeoML
