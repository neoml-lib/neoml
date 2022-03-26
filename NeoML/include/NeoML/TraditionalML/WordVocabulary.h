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

namespace NeoML {

class NEOML_API CWordVocabulary {
public:
	CWordVocabulary() : totalWordsCount( 0 ) {}	
	CWordVocabulary( const CWordVocabulary& other );
	CWordVocabulary& operator=( const CWordVocabulary& other );
	
	// Количество слов в словаре.
	int Size() const { return words.Size(); }
	// Обрезать словарь.
	void RestrictSize( int maxSize );

	// Получить индекс слова в словаре.
	int GetWordIndex( const CString& word ) const;
	// Проверка наличия слова в словаре.
	bool HasWord( const CString& word ) const;
	
	// Получить слово по индексу.
	CString GetWord( int index ) const;
	// Изменить слово в словаре.
	void SetWord( int index, const CString& word );

	// Сколько раз слово встретилось в корпусе?
	long long GetWordUseCount( int index ) const { return words[index].Count; }
	long long GetWordUseCount( const CString& word ) const;
	// Частота слова в корпусе. Сумма частот всех слов словаря <= 1, т.к.
	// в словаре могут быть не все слова корпуса.
	double GetWordFrequency( int index ) const;
	double GetWordFrequency( const CString& word ) const;

	// Добавить слово в словарь.
	// Добавить слово с количеством встреч в корпусе в словарь.
	void AddWord( const CString& word );
	void AddWordWithCount( const CString& word, long long count );
	// Добавить другой словарь.
	void AddVocabulary( const CWordVocabulary& vocabulary );
	// Завершить формирование словаря.
	// minUseCount - минимальное количество встреч слова в корпусе, чтобы оно вошло в словарь.
	void Finalize( long long minUseCount );

	// Очистить словарь.
	void Empty();

	// Сериализация.
	void Serialize( CArchive& archive );

private:
	struct CWordWithCount {
		// Слово.
		CString Word;
		// Сколько оно раз встретилось в корпусе.
		long long Count;

		CWordWithCount() : Count( 0 ) {}
		CWordWithCount( const CString& word, long long count ) : Word( word ), Count( count ) {}

		void Serialize( CArchive& archive );

		bool operator<( const CWordWithCount& other ) const;
		bool operator==( const CWordWithCount& other ) const;
	};

	// Слова в порядке частотности.
	CArray<CWordWithCount> words;
	// слово -> индекс в словаре.
	CMap<CString, int> wordToIndex;
	// Суммарное количество слов в корпусе.
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
