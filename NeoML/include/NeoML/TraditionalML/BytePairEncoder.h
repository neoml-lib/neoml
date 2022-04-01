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

	// Map: pair of neighbour tokens -> set of ids of words containing these pair of tokens.
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

	// Период очистки кеша.
	// Число слов в кеше примерно равно периоду очистки.
	void SetCachePeriod( int _cachePeriod ) const;

	void Build( int size, const CWordDictionary& vocabulary, bool useEOW = true );

	const CWordDictionary& GetTokens() const;
	void UpdateTokens( const CWordDictionary& newVocabulary );

	// Токенизация.
	void Encode( const CString& word, CArray<int>& tokenIds, CArray<int>& offsets ) const;

	int Size() const;

	// Сериализация.
	void Serialize( CArchive& archive );

private:
	// Словарь BPE.
	CWordDictionary tokens;

	class CCache {
	public:
		CCache();
		void SetCachePeriod( int newPeriod );
		bool Request( const CString& word, CArray<int>& bpeEncoding );
		void Add( const CString& word, const CArray<int>& bpeEncoding );

	private:
		struct CEncodedWord {
			// Токены.
			CFastArray<int, 4> TokenIds;
			// Время жизни.
			long long Time;

			CEncodedWord() : Time( 0 ) {}
			CEncodedWord( const CEncodedWord& other ) :
				Time( other.Time ) {
				other.TokenIds.CopyTo( TokenIds );
			}
		};

		// Кеш токенов слов.
		CMap<CString, CEncodedWord> wordCache;
		// Текущее время кеша.
		long long cacheTime;
		int cachePeriod;
	};

	// Cache запросов к кодировщику.
	mutable CCache cache;

	// 1: Добавлена опция для использования StartOfWordToken.
	static const int currentVersion = 1;

	void doInitializeBuild( const CWordDictionary& vocabulary,
		int tokensCount, CBpeIterativeBuilder& builder );
	void createTrainVocabulary( const CWordDictionary& vocabulary,
		CWordDictionary& trainDictionary ) const;
	CString splitWordIntoInitalTokens( const CString& word ) const;
	CString removeSpecialTokens( const CString& word ) const;
	void calculateOffsets( const CArray<int>& tokenIds,
		CArray<int>& offsets ) const;
};

} // namespace NeoML
