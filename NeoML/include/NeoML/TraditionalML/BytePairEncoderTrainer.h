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

// Class that trains byte-pair-encoding.
class NEOML_API CBytePairEncoderTrainer {
public:
	struct CParams {
		// Max size of encoder.
		int MaxSize;
		// Add EoW token to each word.
		bool UseEndOfWordToken;
		// Add SoW token to each word.
		bool UseStartOfWordToken;

		CParams() :
			MaxSize( 50000 ),
			UseEndOfWordToken( true ),
			UseStartOfWordToken( false )
		{}
	};

	CBytePairEncoderTrainer( const CParams& params, const CWordDictionary& dictionary );

	// Trains and returns a fully trained encoder.
	CPtr<IBytePairEncoder> Train();

	// Trains addtional #stepsCount tokens.
	// Returns true if no additional steps can be performed.
	bool TrainSteps( int stepsCount );

	// Returns true if training has been completed.
	bool IsTrainingCompleted() const;

	// Returns encoder consisting of bpe tokens obtained from completed steps.
	CPtr<IBytePairEncoder> GetEncoder() const;

	// Save/load checkpoint
	void Serialize( CArchive& archive );

private:
	// Encoder trainer params.
	CParams params;

	// The number of completed steps.
	int stepsCompletedCount;

	// The dictionary of pairs of neighbour tokens.
	CWordDictionary pairDictionary;
	// The dictionary of bpe tokens.
	CWordDictionary tokensDictionary;

	// Train data.
	CArray<CArray<CString>> trainWords;
	CArray<long long> trainCounts;

	// Map: pair of neighbour tokens -> set of ids of words containing this pair of tokens.
	typedef CMap<CString, CHashTable<int>> CPairReverseIndex;
	CPairReverseIndex reverseIndex;

	CBytePairEncoderTrainer( const CBytePairEncoderTrainer& other ) = delete;
	CBytePairEncoderTrainer& operator=( const CBytePairEncoderTrainer& other ) = delete;

	void createTrainData( const CWordDictionary& dictionary );
	void buildPairDictionary();
	int calcCurrentStepsCount( int requestedIterationsCount ) const;
	bool trainSingleStep();
};

} // namespace NeoML
