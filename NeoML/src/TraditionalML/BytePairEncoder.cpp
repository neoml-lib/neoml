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

// Some special tokens.
static const CString StartOfWordToken( "</w>" );
static const CString EndOfWordToken( "<\\w>" );
static const CString UnknownToken( "<UNK>" );

// Concatenates tokens.
static inline CString mergeTokens( const CString& first, const CString& second )
{
	return first + second;
}

///////////////////////////////////////////////////////////////////////////////

CBpeIterativeBuilder::CBpeIterativeBuilder() :
	iterationsCompletedCount( 0 ),
	totalIterationsCount( 0 )
{
}

void CBpeIterativeBuilder::Initialize( const CArray<CArray<CString>>& trainSplittedWords,
	const CArray<long long>& trainWordCounts, int _totalIterationsCount )
{
	NeoAssert( _totalIterationsCount > 0 );
	NeoAssert( trainSplittedWords.Size() == trainWordCounts.Size() );

	totalIterationsCount = _totalIterationsCount;
	iterationsCompletedCount = 0;

	trainWordCounts.CopyTo( trainCounts );
	trainWords.SetSize( trainSplittedWords.Size() );
	for( int i = 0; i < trainSplittedWords.Size(); i++ ) {
		trainSplittedWords[i].CopyTo( trainWords[i] );
	}
}

bool CBpeIterativeBuilder::IsCompleted() const
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
	trainWords.Serialize( archive );
	trainCounts.Serialize( archive );
	pairDictionary.Serialize( archive );
	archive.Serialize( reverseIndex );
}

// Calculates the number of iterations for the current Run.
// Returned value cannot exceed requestedIterationsCount.
int CBpeIterativeBuilder::calcIterationsCount( int requestedIterationsCount ) const
{
	NeoAssert( requestedIterationsCount > 0 );
	if( IsCompleted() ) {
		return 0;
	}
	const int remainingIterationsCount = totalIterationsCount - iterationsCompletedCount;
	NeoAssert( remainingIterationsCount > 0 );
	return min( requestedIterationsCount, remainingIterationsCount );
}

// Adds token for each char in dictionary and builds dictionary of token pairs (~ neighbour char pairs).
void CBpeIterativeBuilder::buildPairDictionary( CWordDictionary& newTokens )
{
	for( int i = 0; i < trainWords.Size(); i++ ) {
		const CArray<CString>& tokens = trainWords[i];
		NeoAssert( !tokens.IsEmpty() );
		
		const long long count = trainCounts[i];
		NeoAssert( count > 0 );

		for( int j = 0; j < tokens.Size() - 1; j++ ) {
			const CString pair = mergeTokens( tokens[j], tokens[j + 1] );
			pairDictionary.AddWord( pair, count );

			reverseIndex.GetOrCreateValue( pair ).Add( i );
			newTokens.AddWord( tokens[j], count );
		}
		newTokens.AddWord( tokens.Last(), count );
	}

	pairDictionary.Finalize( 1 );
	iterationsCompletedCount = newTokens.Size();
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
		const CArray<CString>& oldTokens = trainWords[id];
		const long long count = trainCounts[id];

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

		CArray<CString> newTokens;
		int indexesToMergeIndex = 0;
		for( int j = 0; j < oldTokens.Size(); j++ ) {
			if( indexesToMergeIndex >= indexesToMerge.Size() ||
				j != indexesToMerge[indexesToMergeIndex] )
			{
				newTokens.Add( oldTokens[j] );
			} else {
				newTokens.Add( mergeTokens( oldTokens[j], oldTokens[j + 1] ) );
				indexesToMergeIndex++;
				j++;
			}

			if( newTokens.Size() > 1 ) {
				// Updating counts for pairs in new word.
				const CString pair = mergeTokens( newTokens[newTokens.Size() - 2], newTokens.Last() );
				reverseIndex.GetOrCreateValue( pair ).Add( id );
				pairDictionary.AddWord( pair, count );
			}
		}

		newTokens.MoveTo( trainWords[id] );
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
	CArray<int>& tokenIds, CArray<int>& tokenLengths )
{
	if( cachePeriod == NotFound ) {
		return false;
	}

	cacheTime++;
	bool success = false;
	if( wordCache.Has( word ) ) {
		CEncodedWord& ids = wordCache.Get( word );
		for( int i = 0; i < ids.TokenIds.Size(); i++ ) {
			tokenIds.Add( ids.TokenIds[i] );
			tokenLengths.Add( ids.TokenLengths[i] );
		}
		ids.Time = cacheTime;
		success = true;
	}

	// Removes items from cache.
	// The item is erased if there has not been a request with the same word since the previous cleanup. 
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
	const CArray<int>& tokenIds, const CArray<int>& tokenLengths )
{
	NeoAssert( !wordCache.Has( word ) );
	NeoAssert( tokenIds.Size() == tokenLengths.Size() );
	CEncodedWord encodedeWord;
	encodedeWord.Time = cacheTime;
	for( int i = 0; i < tokenIds.Size(); i++ ) {
		encodedeWord.TokenIds.Add( tokenIds[i] );
		encodedeWord.TokenLengths.Add( tokenLengths[i] );
	}
	wordCache.Add( word, encodedeWord );
}

///////////////////////////////////////////////////////////////////////////////

CBytePairEncoder::CBytePairEncoder() :
	useEndOfWordToken( true ),
	useStartOfWordToken( false )
{}

CBytePairEncoder::CBytePairEncoder( const CBytePairEncoder& other ) :
	tokens( other.tokens ),
	useEndOfWordToken( other.useEndOfWordToken ),
	useStartOfWordToken( other.useStartOfWordToken )
{
}

CBytePairEncoder& CBytePairEncoder::operator=( const CBytePairEncoder& other )
{
	tokens = other.tokens;
	useEndOfWordToken = other.useEndOfWordToken;
	useStartOfWordToken = other.useStartOfWordToken;
	return *this;
}

void CBytePairEncoder::Build( const CWordDictionary& vocabulary, int size,
	bool _useEndOfWordToken, bool _useStartOfWordToken )
{
	useEndOfWordToken = _useEndOfWordToken;
	useStartOfWordToken = _useStartOfWordToken;

	CBpeIterativeBuilder builder;
	InitializeBuilder( vocabulary, size, builder );

	UpdateTokens( builder.RunTotalIterations() );
}

void CBytePairEncoder::InitializeBuilder( const CWordDictionary& dictionary,
	int tokensCount, CBpeIterativeBuilder& builder )
{
	NeoAssert( tokensCount > 0 );

	CArray<CArray<CString>> trainWords;
	CArray<long long> trainCounts;
	createTrainVocabulary( dictionary, trainWords, trainCounts );

	builder.Initialize( trainWords, trainCounts, tokensCount );
}

void CBytePairEncoder::UpdateTokens( const CWordDictionary& newVocabulary )
{
	tokens.AddDictionary( newVocabulary );
	tokens.Finalize( 1 );
}

void CBytePairEncoder::Encode( const CString& word, CArray<int>& tokenIds,
	CArray<int>& tokenLengths ) const
{
	if( cache.Request( word, tokenIds, tokenLengths ) ) {
		return;
	}

	CArray<CString> wordTokens;
	CArray<int> wordTokenLengths;
	splitWordIntoInitalTokens( word, wordTokens, &wordTokenLengths );

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
		wordTokenLengths[bestMergePos] += wordTokenLengths[bestMergePos + 1];
		
		wordTokens.DeleteAt( bestMergePos + 1 );
		wordTokenLengths.DeleteAt( bestMergePos + 1 );
	}

	NeoAssert( wordTokens.Size() == wordTokenLengths.Size() );
	for( int i = 0; i < wordTokens.Size(); i++ ) {
		tokenIds.Add( tokens.GetWordId( wordTokens[i] ) );
	}
	tokenLengths.Add( wordTokenLengths );

	cache.Add( word, tokenIds, wordTokenLengths );
}

void CBytePairEncoder::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( currentVersion );
	tokens.Serialize( archive );
}

// Prepares input vocabulary for bpe training.
void CBytePairEncoder::createTrainVocabulary( const CWordDictionary& dictionary,
	CArray<CArray<CString>>& trainWords, CArray<long long>& trainCounts ) const
{
	trainWords.SetSize( dictionary.Size() );
	trainCounts.SetSize( dictionary.Size() );
	for( int i = 0; i < dictionary.Size(); i++ ) {
		splitWordIntoInitalTokens( dictionary.GetWord( i ), trainWords[i] );
		trainCounts[i] = dictionary.GetWordUseCount( i );
	}
}

static bool isNewUtf8Symbol( char c )
{
#pragma message( WARNING_PREGFIX "Not Tested" )
	const unsigned char codePoint = static_cast<unsigned char>( c );
	return ( codePoint >> 7 ) == 0
		|| ( codePoint >> 6 ) == 3;
}

void CBytePairEncoder::splitWordIntoInitalTokens( const CString& word,
	CArray<CString>& splittedWord, CArray<int>* initialLengths ) const
{
	NeoAssert( splittedWord.IsEmpty() );
	if( useStartOfWordToken ) {
		splittedWord.Add( StartOfWordToken );
	}

	CString currentToken;
	for( int i = 0; i < word.Length(); i++ ) {
		if( i > 0
			&& isNewUtf8Symbol( word[i] ) )
		{
			NeoAssert( !currentToken.IsEmpty() );
			splittedWord.Add( currentToken );
			currentToken = "";
		}
		currentToken += word[i];
	}
	NeoAssert( !currentToken.IsEmpty() );
	splittedWord.Add( currentToken );
	
	if( useEndOfWordToken ) {
		splittedWord.Add( EndOfWordToken );
	}

	if( initialLengths != nullptr ) {
		NeoAssert( initialLengths->IsEmpty() );
		initialLengths->Add( 1, splittedWord.Size() );
		if( useStartOfWordToken ) {
			initialLengths->First() = 0;
		}
		if( useEndOfWordToken ) {
			initialLengths->Last() = 0;
		}
	}
}

} // namespace NeoML
