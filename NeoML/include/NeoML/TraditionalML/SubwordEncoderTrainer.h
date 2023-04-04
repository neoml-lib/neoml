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

#include <NeoML/TraditionalML/WordDictionary.h>
#include <NeoML/TraditionalML/SubwordEncoder.h>

namespace NeoML {
class CBpeTrainer;

// Class that trains byte-pair-encoding.
class NEOML_API CSubwordEncoderTrainer {
public:
	enum class TAlgorithm {
		BPE,
		Unigram
	};

	enum class TBorderHandling {
		// Add special EndOfWord symbol (</s>) to all words
		EndOfWord,
		// Add special BeginOfWord symbol <s> to all words
		BeginOfWord,
		// Same as BeginOfWord, but with U+2581 as <s>.
		// Note that the encoder has no special mode to handle this option. It is a user responsibility to place U+2581.
		SentencePiece,
		// Add special symbols on both sides of words
		BeginAndEndOfWord,
		// No preprocessing
		None
	};

	enum class TVocabPruning {
		// Restrict a single-letter vocabulary based on their frequency. Default coverage is 1, all symbols will be included into the vocabulary.
		Coverage,
		// Treat training data as raw bytes. Initial vocabulary size is 255, no <UNK> symbols will appear.
		ByteBPE
	};

	CSubwordEncoderTrainer( int vocabSize, TAlgorithm, TBorderHandling, TVocabPruning = TVocabPruning::Coverage );
	~CSubwordEncoderTrainer();

	// Prune single-letter vocabulary so that it covers 'fraction' of the training data. Useful when text contains many rare unicode symbols.
	// By default initial vocabulary contains all found chars (fraction = 1)
	void SetCharacterCoverage( double fraction );
	// Explicitly define required letters that cannot be deleted while pruning
	void SetMandatoryChars( const CArray<CString>& );
	// 0 by default. All other tokens will have contiguous numbers from ( UnknownTokenId + 1 )
	void SetUnknownTokenId( int value );

	// Trains and returns a fully trained encoder.
	CPtr<IBytePairEncoder> Train( const CWordDictionary& frequencyDict );

	CSubwordEncoderTrainer( const CSubwordEncoderTrainer& other ) = delete;
	CSubwordEncoderTrainer& operator=( const CSubwordEncoderTrainer& other ) = delete;

private:
	// Trainer and encoder params
	int desiredVocabSize;
	TBorderHandling borderHandling;
	TVocabPruning vocabPruning;
	double coverage = 1.;
	int encoderUnkTokenId = 0;

	CArray<CString> mandatoryTokens;
	CBpeTrainer* bpeTrainer = nullptr;
	// CPtrOwner<CUnigramTrainer> unigramTrainer;

	CWordDictionary getInitialDictionary( const CWordDictionary& trainData ) const;
	static CWordDictionary getAllBytesDictionary();
};

} // namespace NeoML
