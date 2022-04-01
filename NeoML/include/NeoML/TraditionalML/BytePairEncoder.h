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

class NEOML_API CBpeIterativeBuilder {
public:
	CBpeIterativeBuilder();

	// Инициализация механизма.
	void Initialize( const CWordDictionary& vocabulary, int totalIterationsCount );

	// Вычислить заданное кол-во итераций.
	// Возращается словарь новых токенов.
	CWordDictionary RunIterations( int iterationCount );
	// Вычислить полное кол-во итераций.
	CWordDictionary RunTotalIterations();

	// Общее кол-во завершенных итерация.
	int IterationsCompletedCount() const { return iterationsCompletedCount; }
	// Общее кол-во запланированных итераций.
	int TotalIterationsCount() const { return totalIterationsCount; }
	// Завершено ли построение.
	bool IsBuildCompleted() const;

	void Serialize( CArchive& archive );

private:
	// Выполенное кол-во итераций.
	int iterationsCompletedCount;
	// Общее запланированное кол-во итераций.
	// NotFound - ограничения нет.
	int totalIterationsCount;

	// Текущий словарь словар.
	CWordDictionary trainVocabulary;
	// Текущий словарь пар токенов.
	CWordDictionary pairVocabulary;

	// По паре смердженных токенов хранит информацию об индексах слов в словаре, где эти 
	// пары встречаются.
	typedef CMap<CString, CHashTable<int>> CPairReverseIndex;
	CPairReverseIndex reverseIndex;

	int calcIterationsCount( int requestedIterationsCount ) const;
	void buildPairVocabulary( CWordDictionary& newTokens );
	bool updatePairVocabulary( CWordDictionary& newTokens );
};

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
		CWordDictionary& trainVocabulary ) const;
	CString splitWordIntoInitalTokens( const CString& word ) const;
	CString removeSpecialTokens( const CString& word ) const;
	void calculateOffsets( const CArray<int>& tokenIds,
		CArray<int>& offsets ) const;
};

} // namespace NeoML
