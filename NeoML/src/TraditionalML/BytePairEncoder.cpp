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


#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/BytePairEncoder.h>

namespace NeoML {

static const char* EndOfWordToken = "<EOW>";
static const char* UnknownToken = "<UNK>";
static const char Separator = ' ';

static CString mergeTokens( const CString& first, const CString& second )
{
	return first + second;
}

static void split( const CString& string, CArray<CString>& result )
{
	result.DeleteAll();
	for( int pos = 0; pos <= string.GetLength(); ) {
		int nextDelimeter = string.Find( Separator, pos );
		if( nextDelimeter == NotFound ) {
			nextDelimeter = string.GetLength();
		}
		if( nextDelimeter > pos ) {
			result.Add( CString( ( const char* )string + pos, nextDelimeter - pos ) );
		}
		pos = nextDelimeter + 1;
	}
}

CBpeIterativeBuilder::CBpeIterativeBuilder() :
	iterationsCompletedCount( 0 ),
	totalIterationsCount( NotFound )
{
}

void CBpeIterativeBuilder::Initialize( const CWordVocabulary& vocabulary,
	int _totalIterationsCount )
{
	vocabulary.CopyTo( trainVocabulary );
	totalIterationsCount = _totalIterationsCount;
	assert( totalIterationsCount > 0 );
	iterationsCompletedCount = 0;
}

bool CBpeIterativeBuilder::IsBuildCompleted() const
{
	// ������ ������ ������� ����� ���� ������� (~ ��� ������� ����� ��� ���� ��������� �����).
	const bool isNoMergeAvailable = iterationsCompletedCount > 0
		&& pairVocabulary.Size() == 0;
	// ���������� ����������� �� ����� ����� ��������.
	const bool isTotalRunCountAchieved = iterationsCompletedCount == totalIterationsCount;

	return isNoMergeAvailable
		|| isTotalRunCountAchieved;
}

CWordVocabulary CBpeIterativeBuilder::RunTotalIterations()
{
	return RunIterations( totalIterationsCount );
}

CWordVocabulary CBpeIterativeBuilder::RunIterations( int requestedIterationsCount )
{
	assert( requestedIterationsCount > 0 );
	CWordVocabulary newTokens;

	if( iterationsCompletedCount == 0 ) {
		// �� ������ �������� �������������� ������� �������.
		buildPairVocabulary( newTokens );
	}

	const int iterationsCount = calcIterationsCount( requestedIterationsCount );
	for( int i = 0; i < iterationsCount; i++ ) {
		if( !updatePairVocabulary( newTokens ) ) {
			break;
		}
		iterationsCompletedCount++;
	}

	return newTokens;
}

void CBpeIterativeBuilder::Serialize( CArchive& archive )
{
	archive.Serialize( iterationsCompletedCount );
	archive.Serialize( totalIterationsCount );
	trainVocabulary.Serialize( archive );
	pairVocabulary.Serialize( archive );
	archive.Serialize( reverseIndex );
}

int CBpeIterativeBuilder::calcIterationsCount( int requestedIterationsCount ) const
{
	NeoAssert( requestedIterationsCount > 0 );
	if( IsBuildCompleted() ) {
		return 0;
	}
	const int remainingIterationsCount = totalIterationsCount - iterationsCompletedCount;
	NeoAssert( remainingIterationsCount > 0 );
	return min( requestedIterationsCount, remainingIterationsCount );
}

// ������� ������� ���.
void CBpeIterativeBuilder::buildPairVocabulary( CWordVocabulary& newTokens )
{
	for( int i = 0; i < trainVocabulary.Size(); i++ ) {
		const CString& word = trainVocabulary.GetWord( i );
		const long long count = trainVocabulary.GetWordUseCount( i );

		CArray<CString> tokens;
		split( word, tokens );

		NeoAssert( !tokens.IsEmpty() );
		for( int j = 0; j < tokens.Size() - 1; j++ ) {
			const CString pair = mergeTokens( tokens[j], tokens[j + 1] );
			pairVocabulary.AddWordWithCount( pair, count );
			NeoAssert( count > 0 );

			reverseIndex.GetOrCreateValue( pair ).Add( i );
			newTokens.AddWordWithCount( tokens[j], count );
		}
		newTokens.AddWordWithCount( tokens.Last(), count );
	}

	pairVocabulary.Finalize( 1 );
}

// ���� �������� ���������� ������� BPE
bool CBpeIterativeBuilder::updatePairVocabulary( CWordVocabulary& newPairTokens )
{
	if( pairVocabulary.Size() == 0 ) {
		return false;
	}

	// �������� ������ ���� �� ������� �������������.
	const CString bestPair = pairVocabulary.GetWord( 0 );
	newPairTokens.AddWordWithCount( bestPair, pairVocabulary.GetWordUseCount( 0 ) );

	// ������� ��� id ���� � �������, � ������� ���� ������ ����.
	const CHashTable<int>& wordIdsToChange = reverseIndex.Get( bestPair );
	for( int id : wordIdsToChange ) {
		const CString& word = trainVocabulary.GetWord( id );
		const long long count = trainVocabulary.GetWordUseCount( id );

		// ������ ��������� �� ������.
		CArray<CString> oldTokens;
		split( word, oldTokens );

		// ������� ������� ������ ������� � �����, ������� ����� ���������.
		// ������� �������������� ������� ����������� ���.
		CArray<int> indexesToMerge;
		bool wasPreviousPairMerged = false;
		for( int j = 0; j < oldTokens.Size() - 1; j++ ) {
			const CString pair = mergeTokens( oldTokens[j], oldTokens[j + 1] );
			pairVocabulary.AddWordWithCount( pair, -count );
			if( pair == bestPair ) {
				if( !wasPreviousPairMerged ) {
					indexesToMerge.Add( j );
					wasPreviousPairMerged = true;
					continue;
				}
			} else {
				int wordIndexToDeletePosition = reverseIndex[pair].GetPosition( id );
				if( wordIndexToDeletePosition != NotFound ) {
					reverseIndex[pair].DeleteAt( wordIndexToDeletePosition );
				}
			}
			wasPreviousPairMerged = false;
		}

		NeoAssert( !indexesToMerge.IsEmpty() );

		CString newWord;
		newWord.SetBufferLength( word.Length() );
		int indexesToMergeIndex = 0;
		for( int j = 0; j < oldTokens.Size(); j++ ) {
			if( indexesToMergeIndex >= indexesToMerge.Size() ||
				j != indexesToMerge[indexesToMergeIndex] )
			{
				newWord += oldTokens[j];
			} else {
				newWord += mergeTokens( oldTokens[j], oldTokens[j + 1] );
				indexesToMergeIndex++;
				j++;
			}
			if( j != oldTokens.Size() - 1 ) {
				newWord += Separator;
			}
		}

		trainVocabulary.SetWord( id, newWord );

		CArray<CString> newTokens;
		split( newWord, newTokens );
		for( int j = 0; j < newTokens.Size() - 1; j++ ) {
			const CString pair = mergeTokens( newTokens[j], newTokens[j + 1] );
			reverseIndex.GetOrCreateValue( pair ).Add( id );
			pairVocabulary.AddWordWithCount( pair, count );
		}
	}

	reverseIndex.Delete( bestPair );
	NeoAssert( pairVocabulary.GetWordUseCount( bestPair ) == 0 );
	pairVocabulary.Finalize( 1 );

	return true;
}

///////////////////////////////////////////////////////////////////////////////

class CBpeCache {
public:
	CBpeCache();
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

CBpeCache::CBpeCache()
{
	SetCachePeriod( 1000000 );
}

void CBpeCache::SetCachePeriod( int newPeriod )
{
	NeoAssert( cachePeriod > 0 );
	cachePeriod = newPeriod;
}

bool CBpeCache::Request( const CString& word, CArray<int>& bpeEncoding )
{
	cacheTime++;
	bool success = false;
	if( wordCache.Has( word ) ) {
		CEncodedWord& ids = wordCache.Get( word );
		for( int i = 0; i < ids.TokenIds.Size(); i++ ) {
			bpeEncoding.Add( ids.TokenIds[i] );
		}
		ids.Time = cacheTime;
		success = true;
	}

	// ��� ������� ���� ������� ��������, ���������������� ���� ������ ���� �� ������.
	if( cacheTime % cachePeriod == 0 ) {
		CArray<CString> wordsToDelete;
		for( const CMap<CString, CEncodedWord>::TElement& elem : wordCache ) {
			if( cacheTime - elem.Value.Time >= cachePeriod ) {
				wordsToDelete.Add( elem.Key );
			}
		}

		for( const CString& wordToDelete : wordsToDelete ) {
			wordCache.Delete( wordToDelete );
		}
	}

	return success;
}

void CBpeCache::Add( const CString& word, const CArray<int>& bpeEncoding )
{
	NeoAssert( !wordCache.Has( word ) );
	CEncodedWord encodedeWord;
	encodedeWord.Time = cacheTime;
	for( int tokenId : bpeEncoding ) {
		encodedeWord.TokenIds.Add( tokenId );
	}
	wordCache.Add( word, encodedeWord );
}

///////////////////////////////////////////////////////////////////////////////

CBytePairEncoder::CBytePairEncoder()
{}

CBytePairEncoder::CBytePairEncoder( const CBytePairEncoder& other ) :
	tokens( other.tokens )
{
}

void CBytePairEncoder::SetCachePeriod( int _cachePeriod ) const
{
	if( cache == nullptr ) {
		cache = new CBpeCache();
	}
	cache->SetCachePeriod( _cachePeriod );
}

void CBytePairEncoder::Build( int size, const CWordVocabulary& vocabulary )
{
	NeoAssert( size > 0 );

	CBpeIterativeBuilder builder;
	doInitializeBuild( vocabulary, size, builder );

	UpdateTokens( builder.RunTotalIterations() );
}

void CBytePairEncoder::UpdateTokens( const CWordVocabulary& newVocabulary )
{
	tokens.AddVocabulary( newVocabulary );
	tokens.Finalize( 1 );
}

// ������������� ���� ���������� �����������.
void CBytePairEncoder::doInitializeBuild( const CWordVocabulary& vocabulary,
	int tokensCount, CBpeIterativeBuilder& builder )
{
	// �������������� ������� ��� ��������.
	CWordVocabulary trainVocabulary;
	createTrainVocabulary( vocabulary, trainVocabulary );

	builder.Initialize( trainVocabulary, min( tokensCount, trainVocabulary.Size() ) );
}

void CBytePairEncoder::createTrainVocabulary( const CWordVocabulary& vocabulary,
	CWordVocabulary& trainVocabulary ) const
{
	vocabulary.CopyTo( trainVocabulary );
	for( int i = 0; i < trainVocabulary.Size(); i++ ) {
		const CString& word = trainVocabulary.GetWord( i );
		const CString splittedWord = splitWordIntoInitalTokens( word );
		trainVocabulary.SetWord( i, splittedWord );
	}
}

bool isNewUtf8Symbol( char c )
{
#pragma message( WARNING_PREGFIX "Not Tested" )
	const unsigned char codePoint = static_cast<unsigned char>( c );
	return ( codePoint >> 7 ) == 0
		|| ( codePoint >> 6 ) == 3;
}

CString CBytePairEncoder::splitWordIntoInitalTokens( const CString& word ) const
{
	CString result;
	for( int i = 0; i < word.Length(); i++ ) {
		if( i > 0
			&& isNewUtf8Symbol( word[i] ) )
		{
			result.Append( Separator );
		}
		result.Append( word[i] );
	}
	result.Append( Separator );
	return result + EndOfWordToken;
}

// �����������.
void CBytePairEncoder::Encode( const CString& word, CArray<int>& tokenIds ) const
{
	if( cache != nullptr
		&& cache->Request( word, tokenIds ) )
	{
		return;
	}

	const CString preparedWord = splitWordIntoInitalTokens( word );
	CArray<CString> wordTokens;
	split( preparedWord, wordTokens );

	while( true ) {
		long long bestUseCount = 0;
		int bestMergePos = NotFound;
		for( int i = 0; i < tokens.Size() - 1; i++ ) {
			const CString pair = mergeTokens( wordTokens[i], wordTokens[i + 1] );
			const long long pairUseCount = tokens.GetWordUseCount( pair );
			if( pairUseCount > bestUseCount ) {
				bestUseCount = pairUseCount;
				bestMergePos = i;
			}
		}

		if( bestMergePos == NotFound ) {
			break;
		}

		wordTokens[bestMergePos] = mergeTokens( wordTokens[bestMergePos], 
			wordTokens[bestMergePos + 1] );
		wordTokens.DeleteAt( bestMergePos + 1 );
	}

	for( const CString& token : wordTokens ) {
		tokenIds.Add( tokens.GetWordIndex( token ) );
	}

	if( cache != nullptr ) {
		cache->Add( word, tokenIds );
	}
}

CString CBytePairEncoder::Decode( const CArray<int>& tokenIds ) const
{
	CString result;
	for( int i = 0; i < tokenIds.Size(); i++ ) {
		const int token = tokenIds[i];
		if( token != NotFound ) {
			result.Append( tokens.GetWord( token ) );
		} else {
			result.Append( UnknownToken );
		}
		if( i < tokenIds.Size() - 1 ) {
			result.Append( Separator );
		}
	}
	return result;
}

void CBytePairEncoder::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( currentVersion );
	tokens.Serialize( archive );
}

} // namespace NeoML
