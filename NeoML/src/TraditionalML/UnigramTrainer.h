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
#include <UnigramEncoder.h>

namespace NeoML {
class CUnigramTrainer {
public:
	CUnigramTrainer( int vocabSize, CSubwordEncoderTrainer::TBorderHandling, bool useByteBpe, int unknownTokenId );

	// Trains and returns a fully trained encoder.
	CPtr<IUnigramEncoder> Train( const CWordDictionary& frequencyDict, const CWordDictionary& charVocab );

private:
	struct CTrieCounterData;
	class CTrieCounterComparator;
	struct CTokenLoss;

	// A subword with number of occurrences in training data
	struct CTrainingSubword : public IUnigramEncoder::CSubword {
		CTrainingSubword( CString text, int64_t count );

		int64_t Count = 0;
	};

	using TBorderHandling = CSubwordEncoderTrainer::TBorderHandling;
	using CTokenTrie = CTrieNode<CTrainingSubword*>;

	// Maximal length of tokens
	static constexpr int maxSubwordLength = 10;
	// Number of occurrences in training data required to become a candidate
	static constexpr int minSubwordFreq = 2;
	// Size of the initial vocabulary of candidates (most frequent substrings in train data)
	static constexpr int initialVocabSize = 1000000;
	// Number of best tokenizaions used to evaluate frequencies in E-step
	static constexpr int maxTokenizations = 10;
	// Minimal quality of tokenizations used to evaluate frequencies in E-step
	static constexpr double minQuality = -200;
	// number of EM-algorithm iterations every training step
	static constexpr int nEmIterations = 2;
	// Share of candidates to keep after each training step
	static constexpr double shrinkingFactor = 0.75;
	// Small number to prevent equal scores of forcibly added chars
	static constexpr double scoreEps = 0.0001;

	// train data
	CWordDictionary trainDict;
	// encoder parameters
	ISubwordEncoder::CParams params;
	const int desiredVocabSize;

	CHashTable<CString> chars;
	CTokenTrie candidatesTrie;
	CPointerArray<CTrainingSubword> candidatesStorage;

	int getTokenLength( const CString& str, int pos ) const;
	void fillTrainDict( const CWordDictionary& frequencyDict );
	void createInitialVocab();
	static void dfsTrieFillQueue( CTrieNode<CTrieCounterData>* node,
		CPriorityQueue<CArray<CTrieNode<CTrieCounterData>*>, CTrieCounterComparator>& outQueue );
	bool trainStep();
	void runEmIteration();
	void calcProbsInWord( const CString& word, int64_t count, CMap<CString, double>& probs ) const;
	void dfsUpdateTrieProbs( CTokenTrie* node, const CMap<CString, double>& probs );
	void dfsGetLosses( const CTokenTrie* node, CArray<CTokenLoss>& losses ) const;
	void getTokenLoss( double tokenScore, int64_t tokenCount, CTokenLoss& tokenLoss ) const;
	static void dfsTrieToArray( CTokenTrie* node, CArray<IUnigramEncoder::CSubword>& output );
	void addChars( CArray<IUnigramEncoder::CSubword>& output ) const;
};
} // namespace NeoML
