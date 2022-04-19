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

namespace NeoML {

class NEOML_API CWordVocabulary {
public:
	CWordVocabulary() : totalWordsCount( 0 ) {}	
	CWordVocabulary( const CWordVocabulary& other );
	CWordVocabulary& operator=( const CWordVocabulary& other );
	
	// ���������� ���� � �������.
	int Size() const { return words.Size(); }
	// �������� �������.
	void RestrictSize( int maxSize );

	// �������� ������ ����� � �������.
	int GetWordIndex( const CString& word ) const;
	// �������� ������� ����� � �������.
	bool HasWord( const CString& word ) const;
	
	// �������� ����� �� �������.
	CString GetWord( int index ) const;
	// �������� ����� � �������.
	void SetWord( int index, const CString& word );

	// ������� ��� ����� ����������� � �������?
	long long GetWordUseCount( int index ) const { return words[index].Count; }
	long long GetWordUseCount( const CString& word ) const;
	// ������� ����� � �������. ����� ������ ���� ���� ������� <= 1, �.�.
	// � ������� ����� ���� �� ��� ����� �������.
	double GetWordFrequency( int index ) const;
	double GetWordFrequency( const CString& word ) const;

	// �������� ����� � �������.
	// �������� ����� � ����������� ������ � ������� � �������.
	void AddWord( const CString& word );
	void AddWordWithCount( const CString& word, long long count );
	// �������� ������ �������.
	void AddVocabulary( const CWordVocabulary& vocabulary );
	// ��������� ������������ �������.
	// minUseCount - ����������� ���������� ������ ����� � �������, ����� ��� ����� � �������.
	void Finalize( long long minUseCount );

	// �������� �������.
	void Empty();

	// ������������.
	void Serialize( CArchive& archive );

private:
	struct CWordWithCount {
		// �����.
		CString Word;
		// ������� ��� ��� ����������� � �������.
		long long Count;

		CWordWithCount() : Count( 0 ) {}
		CWordWithCount( const CString& word, long long count ) : Word( word ), Count( count ) {}

		void Serialize( CArchive& archive );

		bool operator<( const CWordWithCount& other ) const;
		bool operator==( const CWordWithCount& other ) const;
	};

	// ����� � ������� �����������.
	CArray<CWordWithCount> words;
	// ����� -> ������ � �������.
	CMap<CString, int> wordToIndex;
	// ��������� ���������� ���� � �������.
	long long totalWordsCount;

	void buildIndex();
	void checkIndex( int index ) const;

	friend CArchive& operator<<( CArchive& archive, const CWordWithCount& value );
	friend CArchive& operator>>( CArchive& archive, CWordWithCount& value );
};

inline CArchive& operator<<( CArchive& archive, const CWordVocabulary::CWordWithCount& value )
{
	const_cast<CWordVocabulary::CWordWithCount&>( value ).Serialize( archive );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, CWordVocabulary::CWordWithCount& value )
{
	value.Serialize( archive );
	return archive;
}

} // namespace NeoML
