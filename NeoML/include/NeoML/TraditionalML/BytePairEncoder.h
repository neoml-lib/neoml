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
#include <NeoML/TraditionalML/WordVocabulary.h>

namespace NeoML {

class NEOML_API CBpeIterativeBuilder {
public:
	CBpeIterativeBuilder();

	// Инициализация механизма.
	void Initialize( const CWordVocabulary& vocabulary, int totalIterationsCount );

	// Вычислить заданное кол-во итераций.
	// Возращается словарь новых токенов.
	CWordVocabulary RunIterations( int iterationCount );
	// Вычислить полное кол-во итераций.
	CWordVocabulary RunTotalIterations();

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
	CWordVocabulary trainVocabulary;
	// Текущий словарь пар токенов.
	CWordVocabulary pairVocabulary;

	// По паре смердженных токенов хранит информацию об индексах слов в словаре, где эти 
	// пары встречаются.
	typedef CMap<CString, CHashTable<int>> CPairReverseIndex;
	CPairReverseIndex reverseIndex;

	int calcIterationsCount( int requestedIterationsCount ) const;
	void buildPairVocabulary( CWordVocabulary& newTokens );
	bool updatePairVocabulary( CWordVocabulary& newTokens );
};

class CBpeCache;

class NEOML_API CBytePairEncoder {
public:
	CBytePairEncoder();
	CBytePairEncoder( const CBytePairEncoder& other );

	// Период очистки кеша. Если NotFound - не использовать кеш.
	// Число слов в кеше примерно равно периоду очистки.
	void SetCachePeriod( int _cachePeriod ) const;

	void Build( int size, const CWordVocabulary& vocabulary );

	void UpdateTokens( const CWordVocabulary& newVocabulary );

	// Токенизация.
	void Encode( const CString& word, CArray<int>& tokenIds ) const;
	CString Decode( const CArray<int>& tokenIds ) const;

	// Сериализация.
	void Serialize( CArchive& archive );

private:
	// Словарь BPE.
	CWordVocabulary tokens;
	// Cache запросов в кодировщику.
	mutable CPtrOwner<CBpeCache> cache;

	// 1: Добавлена опция для использования StartOfWordToken.
	static const int currentVersion = 1;

	void doInitializeBuild( const CWordVocabulary& vocabulary,
		int tokensCount, CBpeIterativeBuilder& builder );
	void createTrainVocabulary( const CWordVocabulary& vocabulary,
		CWordVocabulary& trainVocabulary ) const;
	CString splitWordIntoInitalTokens( const CString& word ) const;
};

} // namespace NeoML
