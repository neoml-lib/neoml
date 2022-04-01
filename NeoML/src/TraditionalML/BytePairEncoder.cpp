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


#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/BytePairEncoder.h>

namespace NeoML {

static const CString EndOfWordToken( "<\\w>" );
static const CString UnknownToken( "<UNK>" );
static const CString Separator = " ";

static CString mergeTokens( const CString& first, const CString& second )
{
	return first + second;
}

static void split( const CString& string, CArray<CString>& result )
{
	result.DeleteAll();
	for( int pos = 0; pos <= string.Length(); ) {
		int nextDelimeter = string.Find( Separator, pos );
		if( nextDelimeter == NotFound ) {
			nextDelimeter = string.Length();
		}
		if( nextDelimeter > pos ) {
			result.Add( CString( ( const char* )string + pos, nextDelimeter - pos ) );
		}
		pos = nextDelimeter + 1;
	}
}

CBpeIterativeBuilder::CBpeIterativeBuilder() :
	iterationsCompletedCount( 0 ),
	totalIterationsCount( 0 )
{
}

void CBpeIterativeBuilder::Initialize( const CWordDictionary& vocabulary,
	int _totalIterationsCount )
{
	trainDictionary = vocabulary;
	totalIterationsCount = _totalIterationsCount;
	assert( totalIterationsCount > 0 );
	iterationsCompletedCount = 0;
}

bool CBpeIterativeBuilder::IsBuildCompleted() const
{
	// No more pairs of neighbour tokens can be added.
	const bool isNoMergeAvailable = iterationsCompletedCount > 0
		&& pairDictionary.Size() == 0;
	// No more iterations can be completed.
	const bool isTotalRunCountAchieved = iterationsCompletedCount >= totalIterationsCount;

	return isNoMergeAvailable
		|| isTotalRunCountAchieved;
}

CWordDictionary CBpeIterativeBuilder::RunTotalIterations()
{
	return RunIterations( totalIterationsCount );
}

CWordDictionary CBpeIterativeBuilder::RunIterations( int requestedIterationsCount )
{
	assert( requestedIterationsCount > 0 );
	CWordDictionary newTokens;

	if( iterationsCompletedCount == 0 ) {
		buildPairDictionary( newTokens );
		if( iterationsCompletedCount > requestedIterationsCount ) {
			// If the number of distinct chars in the word dictionary exceeds requestedIterationCount,
			// no extra tokens should be introduced.
			newTokens.RestrictSize( requestedIterationsCount );
			return newTokens;
		}
	}

	const int iterationsCount = calcIterationsCount( requestedIterationsCount );

	for( int i = 0; i < iterationsCount; i++ ) {
		if( !runSingleIteration( newTokens ) ) {
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
	trainDictionary.Serialize( archive );
	pairDictionary.Serialize( archive );
	archive.Serialize( reverseIndex );
}

// Calculates the number of iterations for the current Run.
// Returned value cannot exceed requestedIterationsCount.
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

// Adds token for each char in dictionary and builds dictionary of token pairs (~ neighbour char pairs).
void CBpeIterativeBuilder::buildPairDictionary( CWordDictionary& newTokens )
{
	for( int i = 0; i < trainDictionary.Size(); i++ ) {
		const CString& word = trainDictionary.GetWord( i );
		const long long count = trainDictionary.GetWordUseCount( i );

		CArray<CString> tokens;
		split( word, tokens );

		NeoAssert( !tokens.IsEmpty() );
		for( int j = 0; j < tokens.Size() - 1; j++ ) {
			const CString pair = mergeTokens( tokens[j], tokens[j + 1] );
			pairDictionary.AddWord( pair, count );
			NeoAssert( count > 0 );

			reverseIndex.GetOrCreateValue( pair ).Add( i );
			newTokens.AddWord( tokens[j], count );
		}
		newTokens.AddWord( tokens.Last(), count );
	}

	pairDictionary.Finalize( 1 );
	iterationsCompletedCount = pairDictionary.Size();
}

// Performs one iteration of the algorithm.
// Adds the most frequent pair of neighbour tokens to newPairTokens.
// If no pair of neighbour tokens can be created returns false.
bool CBpeIterativeBuilder::runSingleIteration( CWordDictionary& newPairTokens )
{
	if( pairDictionary.Size() == 0 ) {
		return false;
	}

	// Selection of the most frequent pair of tokens.
	const CString bestPair = pairDictionary.GetWord( 0 );
	newPairTokens.AddWord( bestPair, pairDictionary.GetWordUseCount( 0 ) );

	// Request for all words containing bestPair. 
	const CHashTable<int>& wordIdsToChange = reverseIndex.Get( bestPair );

	for( THashTablePosition pos = wordIdsToChange.GetFirstPosition(); pos != NotFound;
		pos = wordIdsToChange.GetNextPosition( pos ) ) 
	{
		const int id = wordIdsToChange.GetValue( pos );
		const CString& word = trainDictionary.GetWord( id );
		const long long count = trainDictionary.GetWordUseCount( id );

		// Split the word into tokens.
		CArray<CString> oldTokens;
		split( word, oldTokens );

		// Finding positions in the word where bestPair is located. 
		CArray<int> indexesToMerge;
		bool wasPreviousPairMerged = false;
		for( int j = 0; j < oldTokens.Size() - 1; j++ ) {
			const CString pair = mergeTokens( oldTokens[j], oldTokens[j + 1] );
			// Decreasing count for current pair.
			pairDictionary.AddWord( pair, -count );
			if( pair == bestPair ) {
				if( !wasPreviousPairMerged ) {
					// Careful handling of overlapping bestPairs.
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

		// Creating new word in which all neighbouring bestPairs will be replaced by a new token.
		CString newWord;
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

		trainDictionary.SetWord( id, newWord );

		// Updating counts for pairs in new word. 
		CArray<CString> newTokens;
		split( newWord, newTokens );
		for( int j = 0; j < newTokens.Size() - 1; j++ ) {
			const CString pair = mergeTokens( newTokens[j], newTokens[j + 1] );
			reverseIndex.GetOrCreateValue( pair ).Add( id );
			pairDictionary.AddWord( pair, count );
		}
	}

	reverseIndex.Delete( bestPair );
	NeoAssert( pairDictionary.GetWordUseCount( bestPair ) == 0 );
	// Some pairs may have extincted.
	pairDictionary.Finalize( 1 );

	return true;
}

///////////////////////////////////////////////////////////////////////////////

CBytePairEncoder::CCache::CCache()
{
	SetCachePeriod( 1000000 );
}

void CBytePairEncoder::CCache::SetCachePeriod( int newPeriod )
{
	NeoAssert( newPeriod == NotFound || newPeriod > 0 );
	cachePeriod = newPeriod;
}

bool CBytePairEncoder::CCache::Request( const CString& word, 
	CArray<int>& bpeEncoding )
{
	if( cachePeriod == NotFound ) {
		return false;
	}

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

	// При очистке кеша удаляем элементы, использовавшиеся реже одного раза за период.
	if( cacheTime % cachePeriod == 0 ) {
		CArray<CString> wordsToDelete;
		for( TMapPosition pos = wordCache.GetFirstPosition(); pos != NotFound;
			pos = wordCache.GetNextPosition( pos ) )
		{
			const CEncodedWord& encodedWord = wordCache.GetValue( pos );
			if( cacheTime - encodedWord.Time >= cachePeriod ) {
				wordsToDelete.Add( wordCache.GetKey( pos ) );
			}
		}

		for( int i = 0; i < wordsToDelete.Size(); i++ ) {
			wordCache.Delete( wordsToDelete[i] );
		}
	}

	return success;
}

void CBytePairEncoder::CCache::Add( const CString& word, 
	const CArray<int>& bpeEncoding )
{
	NeoAssert( !wordCache.Has( word ) );
	CEncodedWord encodedeWord;
	encodedeWord.Time = cacheTime;
	for( int i = 0; i < bpeEncoding.Size(); i++ ) {
		encodedeWord.TokenIds.Add( bpeEncoding[i] );
	}
	wordCache.Add( word, encodedeWord );
}

///////////////////////////////////////////////////////////////////////////////

CBytePairEncoder::CBytePairEncoder( const CBytePairEncoder& other ) :
	tokens( other.tokens )
{
}

CBytePairEncoder& CBytePairEncoder::operator=( const CBytePairEncoder& other )
{
	tokens = other.tokens;
	return *this;
}

void CBytePairEncoder::SetCachePeriod( int _cachePeriod ) const
{
	cache.SetCachePeriod( _cachePeriod );
}

void CBytePairEncoder::Build( int size, const CWordDictionary& vocabulary, bool useEOW )
{
	NeoAssert( size > 0 );

	CBpeIterativeBuilder builder;
	doInitializeBuild( vocabulary, size, builder );

	UpdateTokens( builder.RunTotalIterations() );
}

const CWordDictionary& CBytePairEncoder::GetTokens() const
{
	return tokens;
}

void CBytePairEncoder::UpdateTokens( const CWordDictionary& newVocabulary )
{
	tokens.AddDictionary( newVocabulary );
	tokens.Finalize( 1 );
}

// Инициализация фазы построения кодировщика.
void CBytePairEncoder::doInitializeBuild( const CWordDictionary& vocabulary,
	int tokensCount, CBpeIterativeBuilder& builder )
{
	// Подготавливаем слоаарь для обучения.
	CWordDictionary trainDictionary;
	createTrainVocabulary( vocabulary, trainDictionary );

	builder.Initialize( trainDictionary, tokensCount );
}

void CBytePairEncoder::createTrainVocabulary( const CWordDictionary& vocabulary,
	CWordDictionary& trainDictionary ) const
{
	trainDictionary = vocabulary;
	for( int i = 0; i < trainDictionary.Size(); i++ ) {
		const CString& word = trainDictionary.GetWord( i );
		const CString splittedWord = splitWordIntoInitalTokens( word );
		trainDictionary.SetWord( i, splittedWord );
	}
}

static bool isNewUtf8Symbol( char c )
{
#pragma message( WARNING_PREGFIX "Not Tested" )
	const unsigned char codePoint = static_cast<unsigned char>( c );
	return ( codePoint >> 7 ) == 0
		|| ( codePoint >> 6 ) == 3;
}

CString CBytePairEncoder::splitWordIntoInitalTokens( const CString& word ) const
{
	NeoAssert( word.Find( Separator ) == NotFound );
	CString result;
	for( int i = 0; i < word.Length(); i++ ) {
		if( i > 0
			&& isNewUtf8Symbol( word[i] ) )
		{
			result += Separator;
		}
		result += word[i];
	}
	result += Separator;
	result += EndOfWordToken;
	return result;
}

// Токенизация.
void CBytePairEncoder::Encode( const CString& word, CArray<int>& tokenIds,
	CArray<int>& offsets ) const
{
	if( cache.Request( word, tokenIds ) ) {
		return;
	}

	const CString preparedWord = splitWordIntoInitalTokens( word );
	CArray<CString> wordTokens;
	split( preparedWord, wordTokens );

	while( true ) {
		long long bestUseCount = 0;
		int bestMergePos = NotFound;
		for( int i = 0; i < wordTokens.Size() - 1; i++ ) {
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

	for( int i = 0; i < wordTokens.Size(); i++ ) {
		tokenIds.Add( tokens.GetWordId( wordTokens[i] ) );
	}

	cache.Add( word, tokenIds );
}

//CString CBytePairEncoder::Decode( const CArray<int>& tokenIds ) const
//{
//	CString result;
//
//	CString currentWord;
//	for( int i = 0; i < tokenIds.Size(); i++ ) {
//		if( tokenIds[i] == NotFound ) {
//			currentWord += UnknownToken;
//		} else {
//			const CString rawToken = tokens.GetWord( tokenIds[i] );
//			const CString clearToken = removeSpecialTokens( rawToken );
//			currentWord += clearToken;
//
//			if( clearToken.Length() < rawToken.Length() ) {
//				// Значит, встретили конец слова.
//				if( !result.IsEmpty() ) {
//					result += Separator;
//				}
//				result += currentWord;
//				currentWord = "";
//			}
//		}
//	}
//	if( !currentWord.IsEmpty() ) {
//		// Вдруг что-то осталось.
//		if( !result.IsEmpty() ) {
//			result += Separator;
//		}
//		result += currentWord;
//	}
//	return result;
//}

int CBytePairEncoder::Size() const
{
	return tokens.Size();
}

void CBytePairEncoder::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( currentVersion );
	tokens.Serialize( archive );
}

CString CBytePairEncoder::removeSpecialTokens( const CString& word ) const
{
	const int clearLength = word.Length() - EndOfWordToken.Length();
	if( word.Find( EndOfWordToken, clearLength ) != NotFound ) {
		return CString( word, clearLength );
	} else {
		return word;
	}
}

void CBytePairEncoder::calculateOffsets( const CArray<int>& tokenIds,
	CArray<int>& offsets ) const
{

}

} // namespace NeoML
