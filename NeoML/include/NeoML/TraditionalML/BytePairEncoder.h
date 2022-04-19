/* Copyright � 2017-2020 ABBYY Production LLC

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

	// ������������� ���������.
	void Initialize( const CWordDictionary& vocabulary, int totalIterationsCount );

	// ��������� �������� ���-�� ��������.
	// ����������� ������� ����� �������.
	CWordDictionary RunIterations( int iterationCount );
	// ��������� ������ ���-�� ��������.
	CWordDictionary RunTotalIterations();

	// ����� ���-�� ����������� ��������.
	int IterationsCompletedCount() const { return iterationsCompletedCount; }
	// ����� ���-�� ��������������� ��������.
	int TotalIterationsCount() const { return totalIterationsCount; }
	// ��������� �� ����������.
	bool IsBuildCompleted() const;

	void Serialize( CArchive& archive );

private:
	// ���������� ���-�� ��������.
	int iterationsCompletedCount;
	// ����� ��������������� ���-�� ��������.
	// NotFound - ����������� ���.
	int totalIterationsCount;

	// ������� ������� ������.
	CWordDictionary trainVocabulary;
	// ������� ������� ��� �������.
	CWordDictionary pairVocabulary;

	// �� ���� ����������� ������� ������ ���������� �� �������� ���� � �������, ��� ��� 
	// ���� �����������.
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

	// ������ ������� ����.
	// ����� ���� � ���� �������� ����� ������� �������.
	void SetCachePeriod( int _cachePeriod ) const;

	void Build( int size, const CWordDictionary& vocabulary, bool useEOW = true );

	const CWordDictionary& GetTokens() const;
	void UpdateTokens( const CWordDictionary& newVocabulary );

	// �����������.
	void Encode( const CString& word, CArray<int>& tokenIds, CArray<int>& offsets ) const;

	int Size() const;

	// ������������.
	void Serialize( CArchive& archive );

private:
	// ������� BPE.
	CWordDictionary tokens;

	class CCache {
	public:
		CCache();
		void SetCachePeriod( int newPeriod );
		bool Request( const CString& word, CArray<int>& bpeEncoding );
		void Add( const CString& word, const CArray<int>& bpeEncoding );

	private:
		struct CEncodedWord {
			// ������.
			CFastArray<int, 4> TokenIds;
			// ����� �����.
			long long Time;

			CEncodedWord() : Time( 0 ) {}
			CEncodedWord( const CEncodedWord& other ) :
				Time( other.Time ) {
				other.TokenIds.CopyTo( TokenIds );
			}
		};

		// ��� ������� ����.
		CMap<CString, CEncodedWord> wordCache;
		// ������� ����� ����.
		long long cacheTime;
		int cachePeriod;
	};

	// Cache �������� � �����������.
	mutable CCache cache;

	// 1: ��������� ����� ��� ������������� StartOfWordToken.
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
